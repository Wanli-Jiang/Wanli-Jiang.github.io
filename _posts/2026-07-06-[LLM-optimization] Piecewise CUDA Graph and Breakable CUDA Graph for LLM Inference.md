---
title: "Piecewise CUDA Graph and Breakable CUDA Graph for LLM Inference"
date: 2026-07-06
categories: [LLM, optimization]
tags: [LLM, CUDA, CUDA Graph, inference]
description: A technical note on full, piecewise, and breakable CUDA Graphs in modern LLM and VLM inference systems.
---

# 1. The Problem

Large language model inference is full of repeated work. During decode, a server repeatedly asks the GPU to run one more token for the active requests. The request IDs and token values change, but the compute pattern is often similar: normalization, linear projections, attention, MLP, logits, and sampling preparation.

大语言模型推理中有大量重复工作。Decode 阶段，server 会反复让 GPU 为当前 active requests 生成下一个 token。Request ID 和 token value 会变，但计算模式往往相似：normalization、linear projection、attention、MLP、logits，以及 sampling 前的准备。

CUDA Graph is attractive because it can record a sequence of GPU operations and replay it with a single `cudaGraphLaunch`. This removes much of the CPU-side overhead from launching many small kernels one by one.

CUDA Graph 有吸引力，是因为它可以记录一段 GPU operations，并用一次 `cudaGraphLaunch` replay。这样可以减少 CPU 侧逐个 launch 许多小 kernel 的开销。

The difficulty is that LLM serving is not static. A modern serving iteration may contain decode requests, prefill requests, prefix-cache hits, KV cache updates, LoRA adapters, MoE routing, speculative decoding, image tokens, or pipeline-parallel intermediate tensors. One monolithic graph cannot cover all of that.

难点在于 LLM serving 并不是静态的。一次现代 serving iteration 可能同时包含 decode requests、prefill requests、prefix-cache hits、KV cache updates、LoRA adapters、MoE routing、speculative decoding、image tokens，或者 pipeline-parallel intermediate tensors。一个 monolithic graph 不可能覆盖所有这些情况。

The core question of this note is:

这篇笔记讨论的核心问题是：

> How can inference frameworks keep most CUDA Graph performance benefits while still supporting dynamic LLM and VLM serving workloads?

> 推理框架如何在支持动态 LLM/VLM serving workload 的同时，尽量保留 CUDA Graph 的性能收益？

The answer is that modern systems no longer treat CUDA Graph as all-or-nothing. They combine several modes:

答案是：现代系统不再把 CUDA Graph 当成一个“非开即关”的优化，而是组合使用几种模式：

```text
Full CUDA Graph:
    capture the whole forward path, best for stable decode

Piecewise CUDA Graph:
    capture regular graphable regions, run dynamic regions eagerly

Breakable CUDA Graph:
    start one capture flow, insert runtime graph breaks around selected ops

Eager fallback:
    handle rare, large, unsupported, or highly dynamic cases
```

To keep the terminology precise, this note uses only a few abbreviations. **KV cache** means the key/value cache used by attention. **MLP** means the feed-forward block inside a transformer layer. **VLM** means a vision-language model. I spell out **Piecewise CUDA Graph** and **Breakable CUDA Graph** in section titles; when I use "piecewise" or "breakable" later, they refer to those mechanisms.

为了避免术语变成黑话，这篇文章只使用少量缩写。**KV cache** 指 attention 使用的 key/value cache。**MLP** 指 transformer layer 中的 feed-forward block。**VLM** 指 vision-language model。标题中会写全 **Piecewise CUDA Graph** 和 **Breakable CUDA Graph**；后文使用 “piecewise” 或 “breakable” 时，都是指这些机制。

# 2. Why CUDA Graph Helps Decode

Decode can be launch-overhead heavy. Each active request usually contributes one new query token. Many kernels are small, especially layer normalization, residual operations, quantization, activation, and small or fused projection kernels. If the CPU has to launch all of them one by one, CPU overhead becomes visible in time per output token.

Decode 容易受到 launch overhead 影响。每个 active request 通常只贡献一个新的 query token。很多 kernel 都比较小，尤其是 layer normalization、residual operation、quantization、activation，以及小型或融合的 projection kernel。如果 CPU 需要逐个 launch 这些 kernel，CPU overhead 就会体现在 time per output token 里。

Continuous batching makes this pattern repeat. It does not mean every iteration contains the same requests. It means the server repeatedly runs the same kind of decode step over a changing active batch.

Continuous batching 会让这种模式不断重复。它并不是说每个 iteration 都包含相同 request，而是说 server 会在不断变化的 active batch 上反复执行同一类 decode step。

```text
step 100:
    active requests = 29
    choose graph bucket = 32
    replay graph_32

step 101:
    active requests = 31
    choose graph bucket = 32
    replay graph_32

step 102:
    active requests = 28
    choose graph bucket = 32
    replay graph_32
```

CUDA Graph does not care about request identity. It cares about tensor addresses, shapes, launch parameters, and compatible metadata paths. The runtime copies current request data into static buffers, pads to a captured bucket, replays the graph, and slices away dummy output rows.

CUDA Graph 不关心 request identity。它关心 tensor 地址、shape、launch parameter，以及 metadata path 是否兼容。Runtime 会把当前 request data copy 到 static buffer，padding 到 captured bucket，replay graph，然后丢弃 dummy output rows。

```text
real batch size = 27
captured bucket = 32

copy 27 real requests into static buffers
pad 5 dummy rows
replay 32-batch CUDA Graph
use first 27 outputs
```

# 3. CUDA Graph Requirements

CUDA Graph replay trades flexibility for low overhead. Four requirements matter most.

CUDA Graph replay 用灵活性换取低 overhead。最重要的约束有四个。

First, tensor addresses must remain stable. If a kernel was captured reading from address `0x1000`, replay still reads from `0x1000`. If a later iteration creates a new tensor at `0x3000`, the graph does not automatically follow that new tensor.

第一，tensor 地址必须稳定。如果 capture 时某个 kernel 从地址 `0x1000` 读取，那么 replay 时仍然会从 `0x1000` 读取。如果后续 iteration 创建了一个位于 `0x3000` 的新 tensor，graph 不会自动跟随这个新 tensor。

```text
capture time:
    input tensor address  = 0x1000
    output tensor address = 0x2000

bad replay:
    new input tensor address = 0x3000
    CUDA Graph still reads from 0x1000
```

Frameworks solve this with static buffers:

Framework 通常用 static buffer 解决这个问题：

```text
each iteration:
    copy real input -> static_input_buffer
    replay CUDA Graph
    read/slice static_output_buffer
```

Second, shapes and launch parameters are fixed for a captured graph. A graph captured for `[256, 4096]` does not automatically become a graph for `[173, 4096]`. Runtime must either use a different graph or pad to the captured shape.

第二，captured graph 的 shape 和 launch parameter 基本固定。为 `[256, 4096]` capture 的 graph 不会自动变成适用于 `[173, 4096]` 的 graph。Runtime 要么使用另一个 graph，要么 padding 到 captured shape。

```text
real num_tokens = 173
captured buckets = [128, 256, 512, 1024]
choose bucket = 256
pad input from 173 tokens to 256 tokens
replay graph captured for 256 tokens
slice output back to 173 tokens
```

Third, first-time JIT compilation, autotuning, host synchronization, dynamic allocation, and dynamic control flow should not happen inside capture. Triton autotuning or Inductor benchmarking may call synchronization for timing, which is illegal during stream capture.

第三，首次 JIT compilation、autotuning、host synchronization、dynamic allocation 和 dynamic control flow 不应该发生在 capture 内部。Triton autotuning 或 Inductor benchmarking 可能为了计时调用 synchronization，这在 stream capture 期间是非法的。

```text
bad:
    begin CUDA Graph capture
    first call to a Triton kernel
    Triton compiles or autotunes inside capture
    capture may fail

good:
    warm up Triton kernel first
    begin CUDA Graph capture
    capture only stable kernel launch
```

Fourth, one monolithic graph is not enough when every serving iteration has a different token count or batch structure. Decode, prefill, mixed prefill/decode, multimodal, LoRA, and MoE paths do not all share one stable execution pattern.

第四，当每个 serving iteration 的 token count 或 batch structure 都不同时，单一 monolithic graph 不够用。Decode、prefill、mixed prefill/decode、multimodal、LoRA 和 MoE path 并不共享同一个稳定执行模式。

# 4. Solving Dynamic Serving Shapes

The solution is not to make one graph handle every possible iteration. The solution is to map an unbounded dynamic workload into a finite set of stable replay paths.

解决方法不是让一个 graph 处理所有可能 iteration，而是把无限动态的 workload 映射到有限数量的稳定 replay path。

```text
1. shape bucketing + padding
2. full CUDA Graph for stable decode
3. piecewise CUDA Graph for prefill and mixed iterations
4. breakable CUDA Graph for dynamic escape hatches
5. scheduler-side shape control
6. eager fallback for rare cases
```

Shape bucketing reduces infinitely many token counts to a finite list:

Shape bucketing 会把无限多种 token count 收敛到有限列表：

```text
capture buckets = [32, 64, 128, 256, 512, 1024, 2048, 4096]

runtime tokens:
    129  -> bucket 256
    2066 -> bucket 4096
    32   -> bucket 32
    173  -> bucket 256
```

Full graph handles the stable decode path:

Full graph 用来处理稳定的 decode path：

```text
active requests = 29 -> full graph bucket 32
active requests = 31 -> full graph bucket 32
active requests = 35 -> full graph bucket 64
```

Piecewise graph handles prefill and mixed iterations by capturing regular compute and leaving dynamic operations outside the graph:

Piecewise graph 通过 capture 规则计算、把动态操作放在 graph 外，来处理 prefill 和 mixed iteration：

```text
graph segment 0:
    norm + qkv projection on padded token bucket

eager segment:
    attention and KV cache update with real metadata

graph segment 1:
    output projection + MLP on padded token bucket
```

Breakable graph inserts runtime breaks:

Breakable graph 会插入 runtime break：

```text
graph_segment_0.replay()
attention_or_kv_transfer_eager()
graph_segment_1.replay()
```

Schedulers can improve graph hit rate by shaping batches. For example, a 4096-token prefill can be chunked into four 1024-token chunks if that matches captured buckets better.

Scheduler 也可以通过调整 batch shape 提高 graph hit rate。例如，如果 4096-token prefill 不适合当前 bucket，可以切成四个 1024-token chunk。

# 5. Warmup, Capture, Replay

Serving systems usually follow three phases:

Serving system 通常分成三个阶段：

```text
1. warmup
2. capture
3. replay
```

Warmup runs real forward passes, but not for user traffic. Its job is to make first-run behavior happen before capture: Triton compilation, autotune, custom kernel initialization, attention metadata setup, allocator warmup, and workspace allocation.

Warmup 会真正执行 forward pass，但不是为了处理用户请求。它的任务是在 capture 前触发首次运行行为：Triton compilation、autotune、custom kernel initialization、attention metadata setup、allocator warmup 和 workspace allocation。

Capture records stable GPU work:

Capture 会记录稳定的 GPU work：

```text
capture:
    begin CUDA Graph capture
    run forward with static buffers
    launch already-compiled kernels
    launch already-selected Triton / CUDA / CUTLASS kernels
    end CUDA Graph capture
```

Replay is the steady state:

Replay 是 steady state：

```text
replay:
    copy real input -> static buffer
    choose captured graph bucket
    cudaGraphLaunch(graph)
    slice or copy output -> real output
```

## 5.1 Choosing Warmup Shapes

Choosing warmup shapes means choosing which runtime shapes deserve ahead-of-time JIT, autotune, metadata initialization, and graph capture.

选择 warmup shape，本质上是在选择哪些 runtime shape 值得提前执行 JIT、autotune、metadata initialization 和 graph capture。

vLLM is descriptor-driven:

vLLM 是 descriptor-driven：

```text
vLLM descriptor dimensions:
    graph mode: FULL / PIECEWISE
    num_tokens
    num_reqs
    uniform_token_count
    num_active_loras
```

Normal decode and speculative decode differ:

普通 decode 和 speculative decode 不同：

```text
normal decode:
    num_reqs = 32
    uniform_token_count = 1
    num_tokens = 32

speculative decode:
    num_reqs = 32
    num_speculative_tokens = 4
    uniform_token_count = 5
    num_tokens = 160
```

SGLang is runner and bucket driven:

SGLang 是 runner/bucket-driven：

```text
SGLang decode shape checks:
    capture_bs from server_args.cuda_graph_bs or defaults
    bs <= max_running_requests
    bs * num_tokens_per_bs must satisfy attention TP/CP multiple
    compile_bs = capture_bs filtered by torch_compile_max_bs
```

TensorRT-LLM is config driven:

TensorRT-LLM 更 config-driven：

```yaml
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024
```

For TensorRT-LLM piecewise graph, the important list is:

对于 TensorRT-LLM piecewise graph，重要的是：

```text
torch_compile_config.capture_num_tokens
```

A good shape set should be chosen from traffic histograms:

好的 shape set 应该来自实际流量 histogram：

```text
decode batch P50 = 24
decode batch P90 = 56
decode batch P99 = 120

reasonable decode buckets:
    [1, 2, 4, 8, 16, 32, 64, 128]

prefill token P50 = 200
prefill token P90 = 900
prefill token P99 = 1800

reasonable token buckets:
    [128, 256, 512, 1024, 2048]
```

# 6. JIT Kernels, FlashInfer, and CUDA Graph

Triton and FlashInfer look dynamic, but frameworks separate dynamic preparation from graph capture.

Triton 和 FlashInfer 看起来很动态，但 framework 会把动态准备阶段和 graph capture 分开。

```text
dynamic preparation:
    Triton JIT / autotune
    FlashInfer plan
    torch.compile / Inductor codegen
    attention metadata initialization

static execution:
    already-compiled Triton kernels
    FlashInfer run()
    fixed metadata buffers
    fixed input/output addresses
```

For Triton, the first call may compile and autotune. That must happen before capture.

对于 Triton，第一次调用可能会 compile 和 autotune。这必须发生在 capture 之前。

```python
# Bad: first Triton call may JIT/autotune inside capture.
with torch.cuda.graph(graph):
    y = triton_kernel(x)

# Good: warmup first.
for _ in range(3):
    y = triton_kernel(x)
torch.cuda.synchronize()

with torch.cuda.graph(graph):
    y_static = triton_kernel(x_static)
```

For FlashInfer, `plan()` is dynamic and runs outside graph. `run()` is the stable GPU execution that can be captured.

对于 FlashInfer，`plan()` 是动态的，运行在 graph 外。`run()` 是稳定的 GPU execution，可以被 capture。

```python
wrapper.plan(
    kv_indptr_buf,
    kv_indices_buf,
    kv_last_page_len_buf,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)

out.copy_(wrapper.run(q, kv_cache))
torch.cuda.synchronize()

with torch.cuda.graph(graph):
    out.copy_(wrapper.run(q, kv_cache))
```

vLLM prepares FlashInfer decode wrappers per CUDA Graph batch size for pure decode full graphs. SGLang runs FlashInfer autotune during warmup and carefully scopes autotune around actual kernel execution. TensorRT-LLM creates CUDA graph metadata before capture and warns not to reallocate tensors stored inside attention metadata after warmup.

vLLM 会为 pure decode full graph 按 batch size 准备 FlashInfer decode wrapper。SGLang 会在 warmup 中执行 FlashInfer autotune，并把 autotune 范围限制在真正的 kernel execution 上。TensorRT-LLM 会在 capture 前创建 CUDA graph metadata，并要求 warmup 后不要重新分配 attention metadata 内部的 tensor。

# 7. Dummy Requests and Padding

CUDA Graph buckets need dummy data in two places:

CUDA Graph bucket 在两个地方需要 dummy data：

```text
1. capture-time dummy request:
       build a representative fake batch for graph capture

2. runtime padding:
       pad a smaller real batch to an already captured shape
```

Capture-time dummy requests must be shape-correct and metadata-safe. They do not need semantic meaning, but they must not trigger kernel assertions, illegal memory access, or writes into real KV cache slots.

Capture-time dummy request 必须 shape 正确且 metadata 安全。它不需要有真实语义，但不能触发 kernel assertion、非法内存访问，或者写入真实 KV cache slot。

Runtime padding fills dummy rows or dummy tokens:

Runtime padding 会填充 dummy rows 或 dummy tokens：

```python
def prepare_decode_replay(real_batch, padded_bs):
    raw_bs = real_batch.bs

    input_ids.fill_(pad_token_id)
    positions.zero_()
    seq_lens.fill_(1)
    is_padding.fill_(True)

    input_ids[:raw_bs].copy_(real_batch.input_ids)
    positions[:raw_bs].copy_(real_batch.positions)
    seq_lens[:raw_bs].copy_(real_batch.seq_lens)
    slot_mapping[:raw_bs].copy_(real_batch.slot_mapping)
    req_pool_indices[:raw_bs].copy_(real_batch.req_pool_indices)

    is_padding[:raw_bs] = False
```

Padding values must be safe. For example, `slot_mapping = 0` is unsafe if KV cache slot zero belongs to a real request. Use dedicated dummy KV slots or padding masks.

Padding value 必须安全。例如，如果 KV cache slot 0 属于真实 request，那么 `slot_mapping = 0` 就不安全。应该使用 dedicated dummy KV slot 或 padding mask。

# 8. Memory Model

CUDA Graph has memory costs beyond the graph handle itself:

CUDA Graph 的显存开销不只是 graph handle：

```text
extra CUDA Graph memory =
    static input buffers
  + static output buffers
  + graph pool reserved memory
  + graph executable metadata
  + captured intermediate buffers
  + inter-segment buffers
  + eager-op output buffers
  + padding overhead
  + backend/autotune/workspace caches
```

PyTorch's CUDA caching allocator can make memory hard to read from `nvidia-smi`. Deleting a tensor usually returns its block to PyTorch's cache, not immediately to the CUDA driver.

PyTorch 的 CUDA caching allocator 会让 `nvidia-smi` 难以解读。删除 tensor 通常只是把 block 还给 PyTorch cache，而不是立刻还给 CUDA driver。

```text
delete tensor:
    tensor object is gone
    memory block may stay in PyTorch cache

new tensor:
    PyTorch reuses cached block
```

`torch.cuda.empty_cache()` releases unused cached blocks, but it cannot release live tensors, model weights, KV cache, or graph pools still referenced by active graphs.

`torch.cuda.empty_cache()` 可以释放 unused cached block，但不能释放 live tensor、model weights、KV cache，或者仍被 active graph 引用的 graph pool。

For fragmentation-heavy workloads, PyTorch has:

对于 fragmentation 很重的 workload，PyTorch 有：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This changes allocator segment management so unused physical pages are easier to reclaim. It is different from `empty_cache()`: `expandable_segments` changes how memory is organized; `empty_cache()` is a manual release action.

它会改变 allocator 的 segment 管理方式，让 unused physical pages 更容易被回收。它和 `empty_cache()` 不同：`expandable_segments` 改变 memory organization；`empty_cache()` 是手动 release action。

Rough memory sizes:

粗略显存量级：

```text
weights:
    8B bf16  ~= 16 GB total
    70B bf16 ~= 140 GB total, ~17.5 GB/GPU with TP=8

KV cache:
    GQA 8B example ~= 128 KB/token
    MHA 8B example ~= 512 KB/token

hidden buffer:
    256 decode tokens, hidden 8192, fp16  ~= 4 MB
    2048 prefill tokens, hidden 8192      ~= 32 MB

MLP intermediate:
    2048 tokens, intermediate 28672, fp16 ~= 117 MB

logits:
    256 decode tokens, vocab 128k, fp16   ~= 62.5 MB
    2048 prefill tokens, vocab 128k, fp16 ~= 500 MB
```

Full CUDA Graph often costs hundreds of MB to a few GB. Piecewise CUDA Graph can cost more because graph count grows with token buckets and segments. Breakable CUDA Graph cost depends on the number of graph breaks.

Full CUDA Graph 通常额外消耗数百 MB 到数 GB。Piecewise CUDA Graph 可能更高，因为 graph count 会随 token bucket 和 segment 数量增长。Breakable CUDA Graph 的开销取决于 graph break 数量。

# 9. Which Operators Are Graph-Friendly?

The rule is not based on the layer name alone. It depends on whether shape, addresses, metadata, and side effects are stable.

判断标准不只是 layer 名字，而是 shape、地址、metadata 和 side effect 是否稳定。

Usually graph-friendly:

通常比较 graph-friendly：

```text
RMSNorm / LayerNorm
Linear / GEMM
MLP
activation
residual add
simple quant / dequant
fixed-shape RoPE
fixed-resolution ViT block
vision projector
```

Usually graph-unfriendly:

通常 graph-unfriendly：

```text
paged attention
prefill attention
mixed prefill/decode attention
dynamic KV cache update
KV offload / load / transfer
scheduler / continuous batching logic
sampling / logits processor
prefix cache lookup / insert
dynamic multimodal packing
dynamic MoE dispatch / all-to-all
```

Conditionally graphable:

条件性可 capture：

```text
MoE expert compute
LoRA
Mamba / SSM
linear attention
RoPE / MRoPE
quantization / dequantization
ViT image encoder
speculative decoding verify path
```

The common piecewise pattern is:

常见 piecewise 模式是：

```text
capture:
    regular dense tensor compute
    norm / linear / MLP / projector / fixed-shape vision blocks

eager or split point:
    attention metadata
    KV cache layout
    dynamic routing
    runtime communication
    scheduler / sampling
    multimodal packing
```

# 10. Full CUDA Graph

Full CUDA Graph captures the entire forward path. It is most natural for decode, where shapes are stable.

Full CUDA Graph 会 capture 整个 forward path。它最适合 shape 稳定的 decode。

vLLM's full graph is descriptor-driven. The descriptor includes token count, request count, uniform token count, and LoRA state. FlashInfer full graph support is mainly pure decode; mixed prefill/decode often falls back.

vLLM 的 full graph 是 descriptor-driven。Descriptor 包含 token count、request count、uniform token count 和 LoRA state。FlashInfer full graph support 主要用于 pure decode；mixed prefill/decode 通常 fallback。

TensorRT-LLM calls the equivalent path generation-only CUDA Graph. It is configured through `cuda_graph_config.batch_sizes` or `max_batch_size`, and runtime can pad to the nearest captured batch size.

TensorRT-LLM 中对应的路径叫 generation-only CUDA Graph。它通过 `cuda_graph_config.batch_sizes` 或 `max_batch_size` 配置，runtime 可以 padding 到最近的 captured batch size。

SGLang's standard decode CUDA Graph similarly captures whole decode forward for a set of batch sizes and uses binary search to select the nearest captured bucket.

SGLang 的 standard decode CUDA Graph 类似，会为一组 batch size capture 完整 decode forward，并用 binary search 选择最近的 captured bucket。

# 11. Piecewise CUDA Graph

Piecewise CUDA Graph exists because full graph is too rigid for prefill and mixed serving iterations.

Piecewise CUDA Graph 存在的原因，是 full graph 对 prefill 和 mixed serving iteration 来说太刚性。

TensorRT-LLM uses `torch.compile` and assumes attention is the main non-capturable component. Attention runs eagerly and writes in-place into a graph-allocated output buffer.

TensorRT-LLM 使用 `torch.compile`，并假设 attention 是主要 non-capturable component。Attention eager 执行，并 in-place 写入 graph-allocated output buffer。

SGLang uses `torch.compile` with a custom backend. The FX graph is split at registered split ops, split ops run eagerly, and graphable submodules are replaced with CUDA piecewise backends.

SGLang 使用带 custom backend 的 `torch.compile`。FX graph 会在 registered split ops 处被切开，split ops eager 执行，可 graph 的 submodule 被替换成 CUDA piecewise backend。

vLLM historically used `torch.compile` / FX splitting and `CUDAGraphWrapper` for pieces. Newer BCG work tries to reduce dependence on compiler partitioning.

vLLM 之前主要使用 `torch.compile` / FX splitting，并用 `CUDAGraphWrapper` 包住 pieces。新的 BCG 工作则试图减少对 compiler partitioning 的依赖。

# 12. Breakable CUDA Graph

Breakable CUDA Graph inserts graph breaks during capture instead of pre-splitting the FX graph.

Breakable CUDA Graph 不预先切 FX graph，而是在 capture 过程中插入 graph break。

vLLM's core idea is:

vLLM 的核心思想是：

```text
begin graph segment
graphable ops
encounter eager break
end segment
run eager function
begin next segment
```

The captured artifact is:

Captured artifact 是：

```text
[
    graph0.replay,
    eager_fn0,
    graph1.replay,
    eager_fn1,
    graph2.replay,
]
```

SGLang exposes similar concepts through `@eager_on_graph` and `break_graph()`. It can be used for debugging or production escape hatches.

SGLang 通过 `@eager_on_graph` 和 `break_graph()` 暴露类似概念。它可以用于 debug，也可以作为 production escape hatch。

TensorRT-LLM does not currently expose a public BCG mechanism comparable to vLLM or SGLang. Its public design is generation-only full graph plus `torch.compile`-based piecewise graph.

TensorRT-LLM 目前没有公开暴露类似 vLLM 或 SGLang 的 BCG 机制。它的公开设计主要是 generation-only full graph 加基于 `torch.compile` 的 piecewise graph。

# 13. Static Address Stability

The most important correctness invariant is address stability across graph/eager boundaries.

最重要的 correctness invariant 是 graph/eager boundary 之间的地址稳定性。

TensorRT-LLM requires attention to write into a tensor allocated by the preceding graph segment. vLLM's BCG decorator requires eager ops to write into caller-provided output tensors. SGLang's BCG docs similarly describe output writeback.

TensorRT-LLM 要求 attention 写入由前一个 graph segment 分配的 tensor。vLLM 的 BCG decorator 要求 eager op 写入 caller-provided output tensor。SGLang 的 BCG 文档也描述了 output writeback。

```text
graph segment 0:
    owns output buffer at address A

eager attention:
    writes result into address A
    does not return fresh tensor at address B

graph segment 1:
    consumes address A
```

# 14. Framework Summary

vLLM emphasizes descriptor-driven dispatch and fresh attention state for full graph capture. It supports full, piecewise, and experimental breakable graph paths. FlashInfer full graph support is mainly pure decode.

vLLM 强调 descriptor-driven dispatch，以及 full graph capture 时使用 fresh attention state。它支持 full、piecewise 和 experimental breakable graph path。FlashInfer full graph support 主要用于 pure decode。

SGLang emphasizes runner/backend separation. Decode graph, piecewise graph, and breakable graph are handled by phase-specific runners and backends. It also has explicit FlashInfer autotune warmup before graph capture.

SGLang 强调 runner/backend 分层。Decode graph、piecewise graph 和 breakable graph 都由 phase-specific runner 和 backend 处理。它还在 graph capture 前显式执行 FlashInfer autotune warmup。

TensorRT-LLM emphasizes config-driven generation-only graph and `torch.compile` piecewise graph. CUDA Graph padding and batch-size tuning have public performance data.

TensorRT-LLM 强调 config-driven generation-only graph 和 `torch.compile` piecewise graph。CUDA Graph padding 和 batch-size tuning 有公开性能数据。

# 15. Open Questions

## 15.1 When Does Piecewise CUDA Graph Stop Paying Off?

SGLang reports broad prefill speedup ranges, and vLLM public PR data shows large gains in some model-runner scenarios. The remaining question is where the crossover point is: for a given model and hardware, when does compute dominate enough that graph segmentation complexity is no longer worth it?

SGLang 报告了较宽的 prefill speedup 范围，vLLM 的公开 PR 数据也显示某些 model-runner 场景收益很大。剩下的问题是 crossover point 在哪里：对于给定模型和硬件，什么时候计算本身已经占主导，以至于 graph segmentation 的复杂度不再值得？

## 15.2 How Many Segments Are Too Many?

Layer-level splitting often makes sense. Op-level splitting may become too fine-grained. The right metric is not only throughput, but also graph count, graph memory, capture time, and time spent in `cudaGraphLaunch`.

Layer-level splitting 通常合理。Op-level splitting 可能过细。正确指标不只是 throughput，还包括 graph count、graph memory、capture time，以及 `cudaGraphLaunch` 时间占比。

## 15.3 How Should Buckets Be Tuned?

TensorRT-LLM reports that finer batch-size coverage can improve throughput but increases memory and startup time. SGLang uses fine token buckets for small sizes and coarser buckets for larger sizes. The general rule is to use finer buckets for high-frequency shapes and fallback for rare large shapes.

TensorRT-LLM 报告更细的 batch-size coverage 可以提升 throughput，但会增加 memory 和 startup time。SGLang 对小 token count 使用细 bucket，对大 token count 使用粗 bucket。通用规则是对高频 shape 使用细 bucket，对低频大 shape fallback。

## 15.4 Can Breakable CUDA Graph Replace Compiler-Based Piecewise Graph?

BCG avoids dependence on `torch.compile`, but it shifts complexity into runtime capture management. It needs careful handling of graph breaks, eager callable replay, stream joins, weak references, and output writeback.

BCG 避免依赖 `torch.compile`，但把复杂度转移到了 runtime capture management。它需要谨慎处理 graph break、eager callable replay、stream join、weak reference 和 output writeback。

## 15.5 How Should Static Address Invariants Be Enforced?

The invariant is simple: downstream graph segments must read the same addresses captured earlier. The open engineering question is how to make this visible and enforceable through typed output-buffer APIs, debug address checks, custom op schemas, and capture-time assertions.

Invariant 很简单：下游 graph segment 必须读取 capture 时的同一批地址。开放的工程问题是如何通过 typed output-buffer API、debug address check、custom op schema 和 capture-time assertion 让这个约束可见且可强制执行。

# 16. References

1. [TensorRT-LLM: Torch Compile & Piecewise CUDA Graph](https://nvidia.github.io/TensorRT-LLM/latest/features/torch_compile_and_piecewise_cuda_graph.html)
2. [SGLang: Piecewise CUDA Graph](https://docs.sglang.io/docs/advanced_features/piecewise_cuda_graph)
3. [SGLang: Breakable CUDA Graph](https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph)
4. [vLLM API: `vllm.compilation.breakable_cudagraph`](https://docs.vllm.ai/en/latest/api/vllm/compilation/breakable_cudagraph/)
5. [vLLM PR: Experimental Breakable CUDA Graph](https://github.com/vllm-project/vllm/pull/42304)
6. [PyTorch Blog: Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
7. [Fireworks.ai: Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)
8. [PyTorch Blog: Compile to Speed Up Inference on Llama 2](https://pytorch.org/blog/pytorch-compile-to-speed-up-inference/)
9. [vLLM source: `vllm/v1/cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py)
10. [vLLM source: `vllm/v1/worker/gpu/cudagraph_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py)
11. [vLLM source: `vllm/compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py)
12. [TensorRT-LLM source: `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py)
13. [TensorRT-LLM blog: Tuning CUDA Graph Batch Sizes for Higher Output Throughput](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md)
14. [TensorRT-LLM runtime examples: CUDA Graph configuration](https://nvidia.github.io/TensorRT-LLM/latest/examples/llm_runtime.html)
15. [SGLang source: `python/sglang/srt/model_executor/cuda_graph_runner.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py)
16. [SGLang source: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)
17. [SGLang PR: Cuda Graph Runner/Backend Refactor](https://github.com/sgl-project/sglang/pull/23906)
18. [vLLM PR: Enable piecewise and full CUDA graphs for pipeline parallelism](https://github.com/vllm-project/vllm/pull/35162)
19. [vLLM PR: Change default CUDA graph mode to FULL_AND_PIECEWISE](https://github.com/vllm-project/vllm/pull/25444)
20. [TensorRT-LLM Architecture Overview: CUDA Graph padding](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)
21. [SGLang CUDA Graph docs: performance impact and PCG schedule](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph)
22. [SGLang PR: Experimental Breakable Piecewise CUDA Graph](https://github.com/sgl-project/sglang/pull/22218)
