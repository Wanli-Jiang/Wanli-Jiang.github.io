---
title: Optimizing Nemotron v3 Super in TensorRT-LLM
date: 2026-03-16
categories: [LLM, optimization]
tags: [LLM]
description: A work log to record the optimization steps for Nemotron v3 Super in TRTLLM.
---

# 1. Introduction

## 1.1 What is Nemotron v3 Super?

Nemotron 3 Super is a 120 billion total parameter (12 billion active) Mixture-of-Experts (MoE) hybrid Mamba-Attention model developed by NVIDIA. It is designed to deliver strong agentic, reasoning, and conversational capabilities, and is optimized for collaborative agents and high-volume workloads with support for up to 1M context length.

The model architecture is built on several key innovations:

1. **Hybrid Mamba-Attention Architecture**: The 88-layer stack follows a periodic interleaving pattern where Mamba-2 blocks are paired with MoE layers. To enable full-token interaction and long-range information routing, a limited number of self-attention layers are strategically inserted as global "anchors". This design offloads the majority of computation to efficient linear-time Mamba blocks and sparse MoE components while preserving global dependency modeling.
2. **LatentMoE**: A novel hardware-aware MoE architecture that projects tokens into a smaller latent dimension for routing and expert computation. This significantly reduces memory bandwidth costs and all-to-all communication overhead, allowing the model to increase the total number of experts and active experts per token without increasing the inference cost.
3. **Multi-Token Prediction (MTP)**: The model incorporates MTP layers with a shared-weight design across prediction heads. This not only improves overall model quality but also enables native speculative decoding, significantly accelerating inference without requiring a separate draft model.
4. **NVFP4 Quantization**: It is the first model in the Nemotron 3 family to be pre-trained using NVFP4 quantization, maximizing compute efficiency on modern hardware like Blackwell GPUs.


## 1.2 Why Choose TensorRT-LLM?

TensorRT-LLM (TRTLLM) is NVIDIA's open-source library for compiling and optimizing Large Language Models for high-performance inference. For a complex and cutting-edge model like Nemotron v3 Super, TRTLLM is the ideal choice for several reasons:

1. **Native Support for Advanced Hardware and Data Types:** Nemotron v3 Super heavily relies on NVFP4 quantization to maximize throughput on modern architectures like Blackwell. TRTLLM provides the foundational support and low-level APIs to leverage these next-generation data types effectively.
2. **Extensibility for Custom Kernels:** The hybrid Mamba-Attention and LatentMoE architecture requires highly specialized kernels to run efficiently. TRTLLM's architecture allows us to seamlessly integrate custom CUDA and Triton kernels (such as optimized MoE, router, and SSU decode kernels) directly into the execution graph.
3. **Aggressive Kernel Fusion:** To minimize latency and memory bandwidth bottlenecks, we need to fuse multiple operations (like quantization, activation, and normalization). TRTLLM's plugin system and graph rewriting capabilities make it possible to implement and inject these fused kernels easily.
4. **Advanced Decoding Features:** Nemotron v3 Super utilizes Multi-Token Prediction (MTP) for speculative decoding. TRTLLM already possesses a robust speculative decoding infrastructure, making the integration of MTP much more straightforward.

# 2. Functional Support for Nemotron v3 Super

Bringing Nemotron v3 Super to TRTLLM required a series of enabling changes across model support, distributed execution, MoE kernels, and numerical stability. The key PRs are:

1. [NVIDIA/TensorRT-LLM#9261](https://github.com/NVIDIA/TensorRT-LLM/pull/9261) added initial Nemotron v3 Super support in the PyTorch backend.
2. [NVIDIA/TensorRT-LLM#10118](https://github.com/NVIDIA/TensorRT-LLM/pull/10118) enabled multi-GPU execution for Nemotron v3 Super.
3. [NVIDIA/TensorRT-LLM#9358](https://github.com/NVIDIA/TensorRT-LLM/pull/9358) added `NVFP4_quant` support to the CUTLASS MoE backend.
4. [NVIDIA/TensorRT-LLM#11470](https://github.com/NVIDIA/TensorRT-LLM/pull/11470) extended multi-GPU support to the TRTLLM MoE backend.
5. [NVIDIA/TensorRT-LLM#11972](https://github.com/NVIDIA/TensorRT-LLM/pull/11972) introduced stochastic rounding for the Mamba cache.

With these pieces in place, we could use `nsys` to capture API and kernel launch traces, then compare simulated and silicon runs to pinpoint the remaining performance gaps.


# 3. Development Methodology: Simulation and Silicon Runs

When optimizing a model for a cutting-edge architecture like Blackwell, we rely on a dual-pronged methodology using both internal simulation tools and actual Silicon runs. This approach allows us to decouple functional correctness and micro-architecture optimization from macro-level system profiling.

## 3.1 Internal Simulation Tools

Before physical hardware is widely available, we rely on internal simulation and profiling tools to begin kernel development early. These tools allow us to validate correctness and perform initial performance tuning in a controlled environment.

The key advantages of using simulation include:
* **Early Development:** We can write, test, and verify CUDA, PTX, and Triton kernels for next-generation hardware before the physical chips are widely available.
* **Deterministic Debugging:** Simulation provides a deterministic environment, which is crucial when debugging complex numerical issues introduced by aggressive low-precision formats like NVFP4.

The main drawback of simulation is speed—it is significantly slower than real hardware. Therefore, we typically use it for unit-testing individual kernels (like the custom MoE or CausalConv1d kernels) rather than running full end-to-end model inferences.

## 3.2 Silicon Runs

Once kernels are functionally verified and micro-optimized in DLSIM, we move to actual Silicon runs on physical GPUs (e.g., B200/GB200). 

Silicon runs are essential for:
* **Macro-Level Profiling:** Using tools like Nsight Systems (nsys) and Nsight Compute (NCU), we can profile the end-to-end execution graph. This helps identify kernel launch overheads, CPU bottlenecks, and PCIe/NVLink communication latencies.
* **End-to-End Benchmarking:** Real hardware is required to measure the true tokens-per-second throughput and time-to-first-token (TTFT) latency, taking into account thermal dynamics, clock scaling, and system-level interactions.
* **Validating Simulator Assumptions:** We constantly compare the NCU reports from Silicon against simulation traces to ensure our simulated optimizations translate to actual hardware speedups.

By iterating between simulation for surgical kernel optimizations and Silicon for holistic system profiling, we were able to systematically eliminate bottlenecks in the Nemotron v3 Super TRTLLM pipeline.

# 4. Optimization Step 1: Optimizing the MoE Kernel

**The Problem:** 
When profiling the execution traces of Nemotron v3 Super, we found that the Mixture-of-Experts (MoE) layers take up the vast majority of the execution time. Specifically, the `routed_experts` (the sparse experts dynamically selected for each token) were running slower than expected when using the default CUTLASS-based kernels. A major bottleneck was the overhead of quantizing data into the `NVFP4` format during the kernel execution.

**The Solution:** 
To reduce latency and push the maximum throughput higher, we replaced the default implementation with a custom, highly optimized MoE kernel generated via `trtllm-gen` (TensorRT-LLM's internal kernel generator). You can see the integration in this PR: [NVIDIA/TensorRT-LLM#10791](https://github.com/NVIDIA/TensorRT-LLM/pull/10791).

**Under the Hood:** 
The `trtllm-gen` MoE kernels are delivered as pre-compiled, hardware-specific `.cubin` binaries that contain Blackwell-optimized batched GEMM implementations designed specifically for MoE workloads (including built-in token routing).

The key innovation is **epilogue fusion with element-wise activation**. In the baseline implementation, after the batched matrix multiplication (one GEMM per expert), the accumulator results must be written to global memory, read back to apply the `ReLU2` activation (i.e. `relu(x)^2`), and then written/read again for `NVFP4` quantization—each round-trip wasting precious memory bandwidth. The new kernel fuses all of this into the GEMM's epilogue stage, where the entire pipeline operates in GPU registers:

1. **Dequant scaling**: The raw accumulator value is multiplied by a dequantization scale factor, restoring it to the proper numerical range needed for the non-linear activation.
2. **Activation** (`ReLU2`): `relu(x)` is applied, then the result is squared—all in registers.
3. **Block scaling + NVFP4 quantization**: The block-level scaling factors for `NVFP4` are computed and the activated values are quantized.
4. **Store to global memory**: The final quantized output and scaling factors are written out only once.

A subtle but critical design detail is the **separation of scaling factors**. Without an activation, the kernel combines dequantization and quantization into a single scale (`dequantScale * quantScale`). But with a non-linear activation like `ReLU2`, the dequantization must happen *before* the activation (otherwise `relu(x)` operates on incorrectly scaled values), so the scales are split: one carries the dequantization factor (applied pre-activation) and the other carries only the quantization factor (applied post-activation). This separation enables numerically correct epilogue fusion for any supported element-wise activation (GELU, ReLU2, SiLU).


# 5. Optimization Step 2: Optimizing the Router Kernel

**The Problem:** 
In a Mixture-of-Experts model, the "router" acts as a traffic controller, deciding which experts should process each incoming token. Nemotron v3 Super has a massive pool of 512 experts, and the router needs to calculate scores and select the top 22 experts for every single token. Relying on native PyTorch operations to calculate these probabilities, sort them, and extract the top 22 is highly inefficient. PyTorch chains multiple generic operations together to achieve this, which creates significant memory overhead and slows down the prefill and decode phases.

**The Solution:** 
We replaced the native PyTorch routing logic with a custom, highly optimized CUDA kernel ([`noAuxTcKernels.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu)) specifically designed for this Top-K routing task. The initial integration to support 512-expert and top-22 configurations can be seen in this PR: [NVIDIA/TensorRT-LLM#9792](https://github.com/NVIDIA/TensorRT-LLM/pull/9792).

**Under the Hood:** 

## 5.1 The Native PyTorch Baseline

Looking at the non-fused path in [`routing.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe/routing.py) (`Deepseekv3RoutingImpl.noaux_tc` with `is_fused=False`), Nemotron v3 Super takes the simplified single-group path (`n_group=1`, `topk_group=1`). The sequence of operations is:

```python
# Step 1: Sigmoid scoring — launches a CUDA kernel to apply sigmoid to all 512 logits
scores = F.sigmoid(logits)  # shape: [num_tokens, 512]

# Step 2: Bias addition — launches another kernel for elementwise add
scores_with_bias = scores + e_score_correction_bias

# Step 3: Top-K selection — PyTorch's generic topk, which internally launches
#         multiple kernels (partial sort + selection)
_, topk_indices = torch.topk(scores_with_bias, k=22, dim=1)

# Step 4: Gather original scores — launches a gather kernel
topk_values = torch.gather(scores, dim=1, index=topk_indices)

# Step 5: Normalize — launches a reduce-sum kernel
topk_values_sum = torch.sum(topk_values, dim=-1, keepdim=True) + 1e-20

# Step 6: Scale — launches a division + multiply kernel
topk_values = topk_values / topk_values_sum * routed_scaling_factor
```

This chains **at least 6 separate CUDA kernel launches** per batch. Each launch incurs CPU-GPU dispatch overhead, and every intermediate result (`scores`, `scores_with_bias`, `topk_indices`, `topk_values`, `topk_values_sum`) must be written to GPU global memory and read back by the next kernel.

## 5.2 The Fused CUDA Kernel

The custom kernel [`deepseek_v3_topk_kernel`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu) replaces the entire pipeline with **a single kernel launch**. For the Nemotron-specific configuration (512 experts, top-22), it is instantiated with a dedicated template specialization:

```cpp
// Nemotron: 512 experts, max top-22, single group, 4 candidates per thread
deepseek_v3_topk_kernel<InputT, OutputT, IdxT, BiasT,
    /*MaxNumExperts=*/512, /*MaxNumTopExperts=*/22,
    /*UseGroups=*/false, /*MaxNumTopGroups=*/4>
```

The kernel launches **one thread block per token** with **512 threads** (one per expert). Here is how it works step by step:

**Step 1 — Load & Score (registers + shared memory):**
Each thread loads its expert's raw logit from global memory, computes `sigmoid_accurate(score)` (a numerically stable sigmoid via `0.5 * tanh(0.5 * x) + 0.5`), adds the routing bias, and writes both the sigmoid score and the biased score to shared memory. This replaces three separate PyTorch kernels (sigmoid, add, topk input) with a single coalesced global memory read per thread.

**Step 2 — Hierarchical Top-K Selection (registers only):**
Since 512 experts exceeds the warp size of 32, the kernel uses a multi-warp hierarchical reduction:
- **Local reduction:** The 512 threads are organized as 16 warps. Each warp loads 4 candidates per thread (covering 128 experts) from shared memory into registers. Within each warp, a `reduceTopK` call finds the local top-22 using an iterative warp-wide maximum reduction.
- **Merge:** The per-warp top-22 results (from 4 warps × 22 = 88 intermediate candidates) are written to shared memory. A single warp then loads these candidates and performs a final `reduceTopK` to extract the global top-22.

The `reduceTopK` algorithm itself is the core of the optimization ([`moeTopKFuncs.cuh`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh)). It packs each (score, expert_index) pair into a single 64-bit integer using CUB's `TwiddleIn` bit manipulation—so that a standard integer `max` correctly compares float values. The 16-bit index is stored inverted (`65535 - idx`) in the lower bits to break ties deterministically. On Blackwell (SM100+), the warp-wide max uses a hardware `redux.sync.max.u32` PTX instruction for single-cycle reduction instead of the multi-cycle cooperative groups path.

**Step 3 — Normalize & Write (warp-level reduce):**
The winning warp gathers the original sigmoid scores (from shared memory) for the selected 22 experts, computes a warp-level sum via `cg::reduce(warp, score, cg::plus{})`, and normalizes: `finalScore = sigmoidScore * routedScalingFactor / (sum + 1e-20)`. The final `topkValues` and `topkIndices` are written to global memory in a single coalesced store.

## 5.3 Summary: Why This is Faster

| Aspect | Native PyTorch | Fused Kernel |
|--------|---------------|--------------|
| Kernel launches | 6+ per batch | **1** |
| Global memory round-trips | 5+ intermediate tensors | **1 read, 1 write** |
| Top-K algorithm | Generic sort-based `torch.topk` | Warp-level iterative max with `redux.sync.max` |
| Sigmoid computation | Separate kernel | Inline `sigmoid_accurate()` in registers |
| Normalization | Separate sum + divide kernels | Warp-level `cg::reduce` + fused scale |
| Shared memory usage | None (all via global memory) | ~4 KB for 512 float scores |
| Blackwell optimization | None | PDL support + hardware `redux.sync.max.u32` |

# 6. Optimization Step 3: Optimizing the SSU Decode Kernel

**The Problem:** 
During the decoding phase (generating tokens one by one), the State Space Update (SSU) kernel inside the Mamba-2 module emerged as the critical path. If this kernel is slow, the entire token generation process is bottlenecked. The default Triton implementation in TRTLLM was inefficient for this specific workload.

**The Solution:** 
We implemented a highly optimized SSU decode kernel. Instead of writing it exclusively for TRTLLM, we contributed it to [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (a popular library for LLM serving kernels) and then integrated it into TRTLLM. 
- FlashInfer PR: [flashinfer-ai/flashinfer#2301](https://github.com/flashinfer-ai/flashinfer/pull/2301)
- TRTLLM Integration PR: [NVIDIA/TensorRT-LLM#10757](https://github.com/NVIDIA/TensorRT-LLM/pull/10757)

## 6.1 The SSU Math

The Selective State Update for Mamba-2 is the recurrence at the heart of every decode step. For a single head, given the current hidden state `state[d][n]` of shape `(dim, dstate)`, the math is:

```
dA          = exp(A * dt)                          # scalar discretization
new_state   = state * dA + (B * dt) * x            # (dim, dstate) recurrence
out         = sum(new_state * C, axis=dstate) + D*x # (dim,) output projection
if z: out   = out * silu(z)                         # optional gating
```

For Nemotron v3 Super with `nheads=128`, `dim=64`, `dstate=128`, and `ngroups=8`, this is computed independently for every head and every batch element, so the total work per decode step is `batch × 128 heads × 64 × 128` state element updates.

## 6.2 The PyTorch Reference: Identifying Bottlenecks

The canonical PyTorch reference implementation lives in `mamba_ssm.ops.triton.selective_state_update_ref` ([source](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py)). Its core loop is:

```python
dt = dt + dt_bias
dt = F.softplus(dt) if dt_softplus else dt
dA = torch.exp(dt.unsqueeze(-1) * A)            # (batch, nheads, dim, dstate)
B  = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # broadcast groups
C  = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)
dB = dt.unsqueeze(-1) * B.unsqueeze(-2)          # (batch, nheads, dim, dstate)
state.copy_(state * dA + dB * x.unsqueeze(-1))   # in-place state update
out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
if D is not None: out += (x * D).to(out.dtype)
out = (out if z is None else out * F.silu(z)).to(x.dtype)
```

This looks clean, but for a decode step (batch=1 token per sequence), it is catastrophically inefficient:

| Operation | What it does | Memory cost |
|-----------|-------------|-------------|
| `dt.unsqueeze(-1) * A` | Broadcasts scalar `dt` across entire `(dim, dstate)` A matrix | Allocates a full `(batch, nheads, dim, dstate)` temporary tensor in global memory |
| `torch.exp(...)` | Element-wise exp | Another full `(batch, nheads, dim, dstate)` read + write |
| `repeat(B, ...)` | Repeats B across grouped heads | Materializes `(batch, nheads, dstate)` from `(batch, ngroups, dstate)` |
| `state * dA + dB * x` | The actual state update | Reads state + dA + dB + x from global memory, writes new state back |
| `torch.einsum(...)` | Output projection | Reads entire new state again, performs contraction |

For Nemotron v3 Super's dimensions, a **single decode step** creates at least 5 separate global memory round-trips and allocates multiple intermediate tensors of shape `(batch, 128, 64, 128)`. Every intermediate write to HBM and subsequent read costs ~1.5 TB/s of bandwidth on a B200—and with batch sizes of 1-4 during decode, the GPU sits idle waiting for memory.

## 6.3 The Triton Kernel: Better, But Limited

The Triton kernel (`_selective_scan_update_kernel` in the same file) improves on the PyTorch reference by fusing the entire SSU into a single kernel launch:

```python
state = tl.load(state_ptrs, mask=...)
x = tl.load(x_ptrs, mask=...).to(tl.float32)
dt = tl.load(dt_ptrs, mask=...).to(tl.float32)
if HAS_DT_BIAS: dt += tl.load(dt_bias_ptrs, mask=...).to(tl.float32)
if DT_SOFTPLUS: dt = tl.where(dt <= 20.0, softplus(dt), dt)
A = tl.load(A_ptrs, mask=...).to(tl.float32)
dA = tl.exp(A * dt[:, None])
B = tl.load(B_ptrs, mask=...).to(tl.float32)
C = tl.load(C_ptrs, mask=...).to(tl.float32)
dB = B[None, :] * dt[:, None]
state = state * dA + dB * x[:, None]
tl.store(state_ptrs, state, mask=...)
out = tl.sum(state * C[None, :], axis=1)
```

The grid is `(cdiv(dim, BLOCK_SIZE_M), batch, nheads)`, with manually tuned block sizes: `BLOCK_SIZE_M=4` when `dstate=128`. Each program instance loads a small tile of dim-rows and the full dstate dimension. While this eliminates intermediate tensors, it has key limitations:

1. **No TMA support.** Triton cannot use the Tensor Memory Accelerator on Hopper+, so all loads go through the standard L2 path.
2. **No producer-consumer overlap.** All loads and computes happen sequentially within the same thread block—there is no way to pre-fetch the next state tile while computing the current one.
3. **Fixed block size tuning.** The hardcoded `(BLOCK_SIZE_M=4, num_warps=4)` for `dstate=128` is a one-size-fits-all compromise that cannot adapt to batch size or GPU occupancy.

## 6.4 The FlashInfer CUDA Kernels: Three Architecture-Specific Designs

Our FlashInfer implementation (`include/flashinfer/mamba/kernel_selective_state_update_stp.cuh`) provides **three** distinct kernel algorithms, auto-dispatched based on GPU architecture and workload size:

### 6.4.1 Simple Kernel (Pre-Hopper / Small Batches)

The simple kernel (`selective_state_update_kernel_simple`) uses 4 cooperative warps to parallelize data loading, then each warp independently processes a subset of the dim dimension:

**Cooperative warp loading.** Instead of having every thread load all inputs sequentially, the 4 warps divide the work:
- **Warp 0**: Loads `x[dim]` and `state_scale[dim]` via vectorized transactions
- **Warp 1**: Loads `B[dstate]` with 128-bit aligned vector loads (`PackedAligned`)
- **Warp 2**: Loads `z[dim]` (or zeros if gating disabled)
- **Warp 3**: Loads `C[dstate]` with the same aligned pattern

After `__syncthreads()`, each warp processes `rowsPerWarp = DIM / numWarps` rows of the state matrix.

**Vectorized state access.** For the innermost loop, the kernel loads state values using `PackedAligned<state_t, stateLoadSize>` where `stateLoadSize` is computed at compile-time via `getVectorLoadSizeForFullUtilization<state_t, DSTATE>()` to maximize 128-bit transactions without leaving warp lanes idle:

```cpp
constexpr unsigned maxHardwareLoadSize = sizeof(float4) / sizeof(T);
constexpr unsigned maxLogicalLoadSize = (unsigned)DSTATE / warpSize;
return min(maxHardwareLoadSize, maxLogicalLoadSize);
```

**Warp-level reduction.** The output `sum(state * C)` is accumulated per-thread and then reduced using `warpReduceSum`:

```cpp
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int s = warpSize / 2; s > 0; s /= 2) {
        val += __shfl_down_sync(UINT32_MAX, val, s);
    }
    return val;
}
```

This keeps everything in registers—no shared memory atomic or global memory write for the reduction.

**Adaptive dim-tiling.** When `batch * nheads < 2 * num_SMs` (GPU under-occupied), the kernel switches from a `(batch, nheads)` grid to a `(batch, nheads, dim_tiles)` 3D grid, splitting the dim dimension across blocks for better SM occupancy:

```cpp
if (total_blocks < num_sms * 2) {
    int const dim_tiles = (DIM + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(params.batch, params.nheads, dim_tiles);
    // kernel with ROWS_PER_BLOCK=4
} else {
    dim3 grid(params.batch, params.nheads);
    // kernel with ROWS_PER_BLOCK=DIM (no tiling)
}
```

### 6.4.2 Vertical Producer-Consumer Kernel (Hopper / SM90+)

The vertical kernel (`selective_state_update_kernel_producer_consumer_vertical`) is the main performance workhorse on Hopper and later. It uses a fundamentally different execution model—**asynchronous producer-consumer pipelining with TMA**.

**Thread block structure:** 5 warps total = 1 producer warp + 4 consumer warps. The producer and consumers operate concurrently on different data tiles, synchronized by `cuda::barrier`:

```cpp
struct SharedStorageVertical {
    alignas(128) state_t state[numStages][rowsPerStage * dstate];  // multi-buffered state
    input_t x[dim], z[dim], B[dstate], C[dstate];
    float out[dim];
    barrier_t bar_empty[numStages];  // producer waits on these
    barrier_t bar_full[numStages];   // consumers wait on these
    barrier_t bar_consumers;
};
```

**Producer warp (warp 4).** The producer runs on a single lane (`cooperative_groups::invoke_one`) and its sole job is orchestrating TMA data movement through a 3-phase pipeline:

1. **Phase 1 – Read Only (filling the pipeline):** Load the first `numStages=3` state tiles from global memory using `cp_async_bulk_tensor_4d_global_to_shared`. The very first stage additionally piggybacks the `x`, `B`, `C`, `z`, and `state_scale` input vectors onto `bar_full[0]` so consumers receive all inputs before the first compute stage:

   ```cpp
   // Stage 0: load all inputs + first state tile
   sram.bar_empty[0].wait(sram.bar_empty[0].arrive());
   cuda::device::memcpy_async_tx(&sram.x[0], x_global_ptr, aligned_size_t<16>(bytesX), sram.bar_full[0]);
   cuda::device::memcpy_async_tx(&sram.B[0], B_global_ptr, aligned_size_t<16>(bytesB), sram.bar_full[0]);
   // ... C, z, state_scale similarly ...
   cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[0][0], &tensorState,
       0, d, head, batch, sram.bar_full[0]);
   auto _ = barrier_arrive_tx(sram.bar_full[0], 1, bytesState + bytesInputs);
   ```

2. **Phase 2 – Steady State (read + write):** For each tile, the producer writes back the previously-computed state tile via TMA (`cp_async_bulk_tensor_4d_shared_to_global`) and simultaneously loads the next tile:

   ```cpp
   cde::fence_proxy_async_shared_cta();  // unblock async proxy for writeback
   cde::cp_async_bulk_tensor_4d_shared_to_global(&tensorState, 0, d_write, head, batch,
       &sram.state[stage][0]);
   cde::cp_async_bulk_commit_group();
   cde::cp_async_bulk_wait_group_read<0>();
   // Then load next:
   cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, 0,
       d_read, head, batch, sram.bar_full[stage]);
   ```

3. **Phase 3 – Write Only (draining):** The final `numStages` tiles only need writeback.

**Consumer warps (warps 0-3).** The 4 consumer warps iterate over stages, each processing `rowsPerStage / consumerWarps` dim-rows per tile. The synchronization is elegant:

```cpp
for (auto dBegin = 0, stage = 0; dBegin < DIM; dBegin += rowsPerStage, stage = (stage+1) % numStages) {
    sram.bar_full[stage].wait(sram.bar_full[stage].arrive());  // wait for producer

    for (auto dd = warp; dd < rowsPerStage; dd += consumerWarps) {
        // ... load state from sram.state[stage][dd * DSTATE + i] ...
        // ... compute new_state = state * dA + dB * x ...
        // ... accumulate out_value += new_state * C ...
        // ... write new state back to sram.state[stage] ...
    }

    cde::fence_proxy_async_shared_cta();
    auto _ = sram.bar_empty[stage].arrive();  // signal producer: buffer is free
}
```

The key insight is that while consumers compute on stage `k`, the producer is already loading stage `k+1` and writing back stage `k-1` via TMA. **This triple-buffering hides the full global memory latency behind compute**.

**TMA Tensor Map.** The state tensor is described by a `CUtensorMap` built via `cuTensorMapEncodeTiled`, defining a 4D layout `(dstate, dim, nheads, batch)` with tile shape `(dstate, rowsPerStage, 1, 1)`. This enables the hardware to autonomously transfer entire state tiles without CPU intervention.

### 6.4.3 Horizontal Producer-Consumer Kernel (Blackwell / SM100+)

On Blackwell GPUs, the kernel auto-dispatches to a horizontal tiling strategy when the state type is `fp16`/`bf16`. Instead of tiling along the dim axis (vertical), it tiles along the dstate axis (horizontal), loading `stageCols = 2 * 32 / sizeof(state_t)` columns per stage.

The key difference is in thread-to-work mapping: `consumerWarps * warpSize` threads are split across the `DIM` rows, with each thread processing `colsPerStage / lanesPerRow` state columns. This layout is better for smaller state types because it keeps more of the computation in registers.

**Bank-conflict-free shared memory.** A critical detail is the `conflict_free_column` permutation function that avoids shared memory bank conflicts:

```cpp
template <int stateValuesPerBank, int numBanks, int colsPerStage>
__device__ __forceinline__ int conflict_free_column(int group, int baseCol) {
    auto const seq_index = group * colsPerStage + baseCol;
    auto const bankCycle = (seq_index / stateValuesPerBank) / numBanks;
    return (baseCol + stateValuesPerBank * bankCycle) % colsPerStage;
}
```

Without this permutation, threads accessing strided patterns would collide on the same shared memory bank. The function offsets each "round" of 32 banks by one slot, completely eliminating serialization.

## 6.5 Additional Features: Quantized State & Stochastic Rounding

For Nemotron v3 Super with NVFP4 quantization, the SSU state cache can be stored in `int16` with per-dim block scaling (`state_scale`). The FlashInfer kernels implement a 2-pass quantization within the warp:

1. **Forward pass:** Compute `new_state` values, track `new_state_max` via `warpReduceMax`.
2. **Quantize pass:** Compute `encode_scale = max_representable / new_state_max`, multiply all values, convert to int16, and write `decode_scale = 1/encode_scale` alongside.

Additionally, when the state is stored in `fp16`, the kernels support **stochastic rounding** via Philox-4x32 PRNG (`PHILOX_ROUNDS > 0`). Instead of deterministic round-to-nearest, each state element is rounded probabilistically, reducing systematic bias in long sequences:

```cpp
philox_randint4x<PHILOX_ROUNDS>(rand_seed, state_ptr_offset + d * DSTATE + i,
    rand_ints[0], rand_ints[1], rand_ints[2], rand_ints[3]);
rState_ptr[e] = cvt_rs_f16_f32(new_state, rand_ints[e % 4] & 0x1FFFu);
```

## 6.6 Auto-Dispatch Logic

The `invokeSelectiveStateUpdate` function automatically selects the best kernel based on GPU capability and workload:

| Condition | Algorithm | Rationale |
|-----------|-----------|-----------|
| Pre-Hopper (SM < 9.0) | Simple | No TMA hardware |
| Hopper/Blackwell + small batch (`batch*nheads < 2*num_SMs`) | Simple (tiled) | Dim-tiling gives better occupancy than producer-consumer at low parallelism |
| Hopper (SM 9.x) | Vertical | TMA + producer-consumer pipelining |
| Blackwell (SM 10.x) + scaled state (`int16`) | Vertical | Horizontal kernel cannot do 2-pass quantization (tiles are discarded after processing) |
| Blackwell (SM 10.x) + `fp16`/`bf16` state | Horizontal | Better register utilization for small state types |
| Blackwell (SM 10.x) + `fp32` state | Vertical | Vertical slightly faster for larger state elements |

By contributing this to FlashInfer, we solved the problem once and benefited multiple serving frameworks simultaneously (vLLM, SGLang, and TRTLLM).

# 7. Optimization Step 4: Aggressive Kernel Fusion

**The Problem:** 
During the prefill phase (processing the initial prompt), we usually cannot use CUDA graphs because the input batch size and sequence length vary constantly. This means every individual PyTorch operation incurs a small CPU-to-GPU "kernel launch" overhead. When you have hundreds of small operations, these launch overheads and the constant reading/writing to global memory add up to a massive delay.

**The Solution:** 
We aggressively fused cascading element-wise operations and quantization steps into single kernels. You can see the bulk of these fusions in [NVIDIA/TensorRT-LLM#11273](https://github.com/NVIDIA/TensorRT-LLM/pull/11273).

**Under the Hood:** 
By combining multiple operations into one, we only pay the kernel launch overhead once, and we keep the data in the GPU's fast registers instead of writing it to global memory and reading it right back. Profiling the code before and after PR #11273 revealed that we completely eliminated over 1,400ms of overhead just by removing redundant memory copy kernels (e.g., `elementwise_kernel (direct_copy)`) and standalone activation kernels. We applied this in two main areas:

## 7.1 Element-wise kernel fusion

In the Mamba-2 mixer, preparing the input for the 1D convolution required slicing and reshaping tensors. In native PyTorch, this forces an explicit and expensive `.contiguous()` memory copy. We wrote custom Triton kernels (like `fuse_elementwise_ops.py` and `fused_split_rearrange_after_conv1d`) that perform the math and place the elements in a naturally contiguous layout in memory. 
* **Result:** For a 50K sequence length, this reduced the time of these operations from ~2.0ms down to just 0.3ms!

## 7.2 NVFP4 quant fusion

To get the massive speedups of NVFP4 matrix multiplications (GEMMs), the input data (which flows through the model in `bf16`) must first be quantized to `NVFP4`. If we do this as a separate step, it wastes time. We solved this by absorbing the quantization step into the operations that happen right before the GEMMs:

### 7.2.1 RMSNorm + NVFP4_quant fusion
For Nemotron v3 Super, the layer pattern is `RMSNorm -> Mixer -> MoE/MLP`. We fused the RMSNorm and NVFP4 quantization into a single CUDA kernel (`FusedAddRMSNormKernel`). The normalization happens, and the result is immediately quantized before being written to memory. This completely eliminated the standalone `flashinfer RMSNormKernel` (saving ~162ms in our profile).

### 7.2.2 Relu2 + NVFP4_quant fusion
The shared expert path in the MoE module is built by `GEMM1 -> ReLU2 -> GEMM2`. We introduced `fusedRelu2QuantizeKernel` which fuses the `ReLU2` activation and the `NVFP4` quantization for GEMM2 into one step. This eliminated the standalone `bias_bf16_relu` kernel, saving an additional ~164ms.

### 7.2.3 Gated_LayerNorm + NVFP4_quant fusion
Following the same pattern, we implemented a custom CUDA version of the non-standard Gated LayerNorm used in the Mamba-2 mixer, fusing it directly with NVFP4 quantization ([PR #11473](https://github.com/NVIDIA/TensorRT-LLM/pull/11473)).


# 8. Optimization Step 5: CausalConv1d Optimization

**The Problem:** 
When analyzing the performance with Nsight Compute (NCU), we discovered that the `causal_conv1d_fwd_kernel` was suffering from severe shared memory bank conflicts. This happens when multiple GPU threads try to access different addresses that map to the same memory bank simultaneously, forcing the hardware to serialize the requests. The original kernel relied entirely on shared memory (`smem_exchange`) for inter-thread data exchange during the sliding-window convolution, creating a serialization bottleneck at every chunk boundary.

**The Solution:** 
We rewrote the core data-exchange path in `causalConv1d.cu`, replaced standard math with hardware intrinsics, added ILP (instruction-level parallelism) through 2-way loop unrolling, and introduced two new fused Triton kernels to eliminate standalone transpose operations. (See [PR #11273](https://github.com/NVIDIA/TensorRT-LLM/pull/11273)).

**Under the Hood:**

### 8.1 Warp shuffle replacing shared memory exchange

To understand this optimization, we first need to understand the kernel's data flow. The `causal_conv1d_fwd_kernel` processes the input sequence in chunks of `kChunkSize = kNThreads * kNElts` elements. Each thread loads a 16-byte vector (`vec_t`, containing `kNElts` elements) from global memory. Because causal convolution requires a sliding window that crosses thread boundaries, every thread needs data from its left neighbor — specifically, the upper half of the previous thread's vector.

In the **original kernel**, this neighbor exchange was done entirely through shared memory: every thread wrote its upper half to `smem_exchange[tidx]`, called `__syncthreads()`, then every thread read from `smem_exchange[tidx - 1]`. The problem is that on NVIDIA GPUs, shared memory is divided into 32 banks. When consecutive threads read from consecutive `smem_exchange` slots, and those slots happen to map to the same bank (which occurs regularly with 16-byte-wide `vec_t` accesses), the hardware must serialize the reads. NCU confirmed that this pattern was causing severe multi-way bank conflicts, with some accesses serialized 4-8x.

Our **rewrite** exploits the fact that within a warp (32 consecutive threads), the "read from the left neighbor" pattern maps perfectly to `__shfl_up_sync` — a warp shuffle instruction that transfers a register value from lane `N-1` to lane `N` in a single cycle, entirely through the register file with zero shared memory involvement. The key subtlety is that `__shfl_up_sync` operates on 32-bit values, but our `vec_t` is 128 bits (16 bytes, enforced by `static_assert(sizeof(vec_t) == 16)`). So we reinterpret-cast the `vec_t` into four `uint32_t` words and issue four independent shuffles. This costs 4 register-to-register moves (each a single-cycle instruction) versus the old path of shared memory write → `__syncthreads()` barrier → shared memory read (tens of cycles with bank conflicts).

The only thread that cannot use warp shuffle is lane 0 of each warp, because its left neighbor lives in the previous warp and warp shuffles cannot cross warp boundaries. For this single thread per warp, we fall back to shared memory — but now only 1 out of 32 threads touches `smem_exchange` for reads, reducing the bank-conflict pressure by roughly 32x. The shared memory writes are still done by all threads (for the benefit of those lane-0 reads), but writes are inherently less conflict-prone than reads because the hardware can buffer them.

### 8.2 Read-only data cache for weight loading

The convolution weight array (`weight[kWidth]`, typically just 4 values for `kWidth = 4`) is constant for the entire kernel execution and is broadcast-read by every thread at the same offsets. The original code loaded weights via a plain global memory dereference, which routes through the L1 cache. The problem is that the L1 cache on NVIDIA GPUs is unified with shared memory (they share the same on-chip SRAM, with a configurable partition). During the heavy shared memory traffic of the neighbor exchange, the L1 portion is already under pressure — adding weight loads to the same path creates additional cache-line evictions and contention.

We changed the weight loads to use `__ldg()` (load via read-only data cache), which routes the memory fetch through a completely separate hardware path — the texture/constant cache. This cache is physically distinct from the L1/shared memory partition and is specifically optimized for broadcast-read patterns where many threads load the same address. Since all threads in a block load the same 4 weight values, this maps perfectly to the read-only cache's design: one cache-line fetch serves all threads, with no interference with the shared memory traffic. For our kernel, this is a strict improvement — no downside, and it frees L1 capacity for the more latency-sensitive `x_vals` loads.

### 8.3 ILP through 2-way unrolled convolution with `__fmaf_rn`

The original convolution inner loop computed one output element at a time: for each element `i`, it iterated over the `kWidth` filter taps, accumulating `weight_vals[w] * x_vals[...]` into `out_vals[i]` using a standard `+=` (which the compiler implements as a separate multiply and add). This creates a long dependency chain — each FMA depends on the previous one's result, and the GPU cannot issue the next instruction until the accumulator is ready (typically 4-8 cycles of arithmetic latency on modern NVIDIA SMs).

Our rewrite changes two things simultaneously:

First, we **replaced `+=` with `__fmaf_rn`** (fused multiply-add, round-to-nearest). On NVIDIA GPUs, a standard `a * b + c` compiles to two instructions (FMUL + FADD), and the intermediate product is rounded before the addition, introducing a small rounding error. `__fmaf_rn` compiles to a single FFMA instruction — it computes `a * b + c` with only one rounding at the end, giving both better throughput (1 instruction instead of 2) and better numerical accuracy (no intermediate rounding).

Second, we **unrolled the loop by a factor of 2**, computing two independent outputs (`acc0`, `acc1`) in each iteration. This is critical for instruction-level parallelism (ILP). With a single accumulator, the warp scheduler must wait for the FFMA result (let's say 4 cycles) before issuing the next FFMA — the functional unit sits idle for those cycles. With two independent accumulators, the scheduler can issue `acc0`'s FFMA, then immediately issue `acc1`'s FFMA on the next cycle (since `acc1` has no data dependency on `acc0`). By the time both are done, `acc0`'s result is ready for the next iteration. This effectively hides the arithmetic latency and keeps the FMA unit near full utilization. We intentionally chose a factor of 2 (rather than 4 or 8) because higher unroll factors increase register pressure, which can reduce occupancy and hurt performance on register-limited kernels.

### 8.4 Fast-math SiLU activation

The CausalConv1d kernel optionally applies SiLU activation (`silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`) to each output element. The original implementation used `expf()`, which is the IEEE-754 compliant exponential function. On NVIDIA GPUs, `expf` is implemented as a multi-instruction software sequence that calls the SFU (Special Function Unit) but then applies range reduction, polynomial correction, and rounding fixups to achieve full single-precision accuracy (~1 ULP error). This sequence typically costs 8-12 instructions.

We replaced it with two hardware intrinsics:
- **`__expf()`**: This maps directly to a single SFU instruction with no software correction. It has ~2 ULP of error (versus ~1 ULP for `expf`), meaning the result can differ by at most 2 units in the last place of the mantissa — completely negligible for inference activations where the output will be immediately consumed by downstream layers. The throughput improvement is roughly 4x because we go from 8-12 instructions down to essentially 1 SFU operation plus overhead.
- **`__frcp_rn()`**: This replaces the division `x / (1 + exp(-x))` with a multiplication `x * (1 / (1 + exp(-x)))`. Division on GPUs is implemented as reciprocal + multiply internally, but the compiler's generic division may insert additional Newton-Raphson refinement steps for accuracy. `__frcp_rn` explicitly uses the SFU's single-instruction reciprocal with round-to-nearest, avoiding any refinement overhead.

The same 2-way unrolling from section 8.3 is applied here as well — we compute SiLU for two elements per iteration, keeping the SFU pipeline busy. Since the SFU is a separate functional unit from the FMA units, SiLU computation doesn't compete with the convolution arithmetic in the preceding loop, further improving overall throughput.

# 9. Optimization Step 6: Tuning Triton Configurations

**The Problem:** 
During the Mamba SSM prefill stage, the model has to process massive input sequences (our target was 50K tokens). This creates immense memory access stress. Nsight Systems (nsys) profiles showed that the State Space Duality (SSD) Triton kernels were underperforming because their block sizes and thread counts were not tuned for this specific workload.

Specifically, the two heaviest kernels in the SSD forward pass are:
- `_chunk_scan_fwd_kernel`: Performs the intra-chunk recurrent scan, iterating over time steps within each chunk to produce the output states. This is the most compute-intensive kernel because it materializes the full `(chunk_size, chunk_size)` attention-like matrix per head.
- `_chunk_state_fwd_kernel`: Computes the inter-chunk hidden state transitions by accumulating the `B * x` outer products across all time steps within each chunk. It acts as the "bridge" that carries state information between consecutive chunks.

The original Triton autotune configurations shipped with the Mamba-2 reference implementation were tuned for older architectures (A100/H100). For instance, the default config for `_chunk_scan_fwd_kernel` looked like:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
                       num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32},
                       num_stages=4, num_warps=4),
    ],
    key=['chunk_size', 'nheads', 'dim'],
)
@triton.jit
def _chunk_scan_fwd_kernel(...):
    ...
```

These configs were biased toward large `BLOCK_SIZE_N` (256) and relatively small `BLOCK_SIZE_K` (32–64), a pattern optimized for H100's L2 cache and shared memory layout. On Blackwell, these configurations led to suboptimal occupancy and excessive register pressure.

**The Solution:** 
Instead of manually guessing the best configurations, I built a standalone benchmarking harness that isolates each SSD kernel and sweeps across the full configuration space. The harness generates representative input tensors matching the exact shapes from our 50K-token prefill workload, then profiles each candidate configuration:

```python
def benchmark_chunk_scan_configs(
    batch_size: int,
    seqlen: int,
    chunk_size: int,
    nheads: int,
    dim: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """Profile _chunk_scan_fwd_kernel across all candidate autotune configs."""

    cb, x, dt, dA_cumsum, C, prev_states = generate_ssd_inputs(
        batch_size, seqlen, chunk_size, nheads, dim, dtype
    )

    nchunks = seqlen // chunk_size

    CANDIDATE_CONFIGS = [
        # Blackwell-oriented: smaller tiles, higher warp counts
        {'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 4, 'num_stages': 4},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 4},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 5},
        # Reference configs from upstream
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_warps': 8, 'num_stages': 3},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 4},
    ]

    results = []
    for cfg in CANDIDATE_CONFIGS:
        grid = (triton.cdiv(chunk_size, cfg['BLOCK_SIZE_M']) *
                triton.cdiv(dim, cfg['BLOCK_SIZE_N']),
                nchunks, batch_size * nheads)

        try:
            latency_ms = triton.testing.do_bench(
                lambda: _chunk_scan_fwd_kernel[grid](
                    cb, x, dt, dA_cumsum, C, prev_states,
                    chunk_size=chunk_size, dim=dim, nheads=nheads,
                    BLOCK_SIZE_M=cfg['BLOCK_SIZE_M'],
                    BLOCK_SIZE_N=cfg['BLOCK_SIZE_N'],
                    BLOCK_SIZE_K=cfg['BLOCK_SIZE_K'],
                    num_warps=cfg['num_warps'],
                    num_stages=cfg['num_stages'],
                ),
                warmup=50, rep=200,
            )
        except triton.OutOfResources:
            latency_ms = float('inf')

        output = _chunk_scan_fwd_kernel[grid](...)
        max_abs_err = (output - reference_output).abs().max().item()

        results.append({**cfg, 'latency_ms': latency_ms, 'max_abs_err': max_abs_err})

    return sorted(results, key=lambda r: r['latency_ms'])
```

I wrapped this in a loop over both kernels, filtered out configs with numerical errors above a tolerance threshold (`max_abs_err < 1e-2` for bf16), and selected the Pareto-optimal configs—fastest latency that still preserves accuracy.

**Under the Hood:** 
The winning configuration for `_chunk_scan_fwd_kernel` on Blackwell was:

```python
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64},
               num_stages=3, num_warps=8)
```

versus the upstream default which selected `BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64`. The reason this smaller tile wins on Blackwell comes down to three factors:

1. **Register pressure and occupancy trade-off.** Each thread block must hold the entire `BLOCK_SIZE_M × BLOCK_SIZE_N` tile of accumulators in registers. With `128 × 256 = 32,768` elements, the register file per SM is nearly saturated, which caps occupancy at 1–2 blocks per SM. Shrinking to `64 × 64 = 4,096` elements frees up registers, allowing 4+ concurrent blocks per SM. On Blackwell's wider SMs with 256 KB register files, higher occupancy translates directly to better latency hiding for the memory-intensive `dA_cumsum` and `C` loads.

2. **Software pipelining depth (`num_stages`).** The `num_stages` parameter controls how many tiles the kernel pre-fetches asynchronously via `cp.async` (or TMA on SM90+). With fewer registers consumed by accumulators, we could afford 3 stages of software pipelining without spilling to local memory. This keeps the memory pipeline fully saturated: while one tile is being computed, the next two are already in flight from global memory to shared memory.

3. **Warp-level parallelism (`num_warps=8`).** With 8 warps per block (256 threads), the scheduler has more warp-level parallelism to hide instruction latencies. Combined with the smaller tile, each warp handles a `64/8 = 8`-row slice of the M dimension, which maps well to Blackwell's warp scheduler that can issue from 4 eligible warps per cycle.

The net effect on `_chunk_scan_fwd_kernel` was a **29% reduction in wall-clock time** across 320 kernel instances (one per layer per chunk). For `_chunk_state_fwd_kernel`, the improvement was more modest (3.6%) because this kernel is more memory-bound (dominated by the outer-product accumulation), so tile-size tuning has less leverage.

| Kernel | Base (ms) | Opt (ms) | Change (ms) | % Change | Instances |
|--------|-----------|----------|-------------|----------|-----------|
| **_chunk_scan_fwd_kernel** | 670.56 | 476.10 | -194.46 | -29.0% | 320 |
| **_chunk_state_fwd_kernel** | 323.61 | 312.06 | -11.55 | -3.6% | 320 |

One important subtlety: Triton's built-in `@triton.autotune` only runs the sweep at the first kernel invocation and caches the winner. In a production serving pipeline where the sequence length varies (e.g., 1K vs 50K), a single cached config can be suboptimal for some shapes. We addressed this by partitioning the autotune key on `chunk_size` and `dim`, so Triton maintains separate cached winners for each unique problem shape. This ensures the kernel always runs with the best config for the current workload without paying repeated autotuning overhead.


# 10. Optimization Step 7: Multi-Token Prediction (MTP) Support

**The Problem:** 
In standard autoregressive decoding, the model generates one token at a time. For latency-sensitive applications, this step-by-step generation is often too slow because the GPU spends most of its time waiting for memory reads rather than doing math (it is memory-bandwidth bound). 

**The Solution:** 
We implemented support for Multi-Token Prediction (MTP) speculative decoding in TRTLLM for Nemotron v3 Super. You can see the integration in this PR: [NVIDIA/TensorRT-LLM#10754](https://github.com/NVIDIA/TensorRT-LLM/pull/10754).

**Under the Hood:** 
MTP allows the model to predict multiple future tokens simultaneously. The model acts as its own "draft" model: lightweight MTP layers (sharing the backbone's weights) guess the next few tokens in a single forward pass. A verification step then checks which guesses are correct and accepts them all at once. Because verifying multiple tokens takes roughly the same time as generating a single one, this dramatically increases tokens-per-second throughput and reduces latency for the end user.

What makes this implementation particularly challenging is that Nemotron v3 Super is a **hybrid Mamba-Attention** model. Unlike pure transformer models where speculative decoding only needs to manage a KV cache, our model has **stateful SSM (state space model) recurrence** and **convolutional state windows** that must be correctly tracked, branched, and rolled back for every speculative draft-and-verify cycle. This required changes across 17 files and ~2,500 lines of code spanning the model architecture, Mamba mixer, state update kernels, Triton convolution kernels, and cache management.

## 10.1 Architecture: The One-Engine MTP Design

Rather than deploying a separate, smaller "draft model" alongside the main model (two-engine speculative decoding), Nemotron v3 Super uses a **one-engine** design. The MTP prediction layers are appended directly to the backbone and share the same execution graph:

```python
class NemotronHForCausalLM(SpecDecOneEngineForCausalLM):
    def __init__(self, model_config):
        super().__init__(
            model=NemotronHModel(model_config),
            model_config=model_config,
        )
        if model_config.spec_config.spec_dec_mode.is_mtp_one_model():
            # Append MTP layers directly to the backbone's layer list
            self.model.layers.extend(self.draft_model.mtp_layers)
            self.epilogue.extend(self.draft_model.mtp_layers)
            self.epilogue.append(self.spec_worker)
```

The base class changes from the standard causal LM to `SpecDecOneEngineForCausalLM`, which provides the speculative scheduling, verification, and token acceptance infrastructure. The backbone forward pass is limited to the first `num_hidden_layers` layers, while the appended MTP layers only run during speculative draft generation.

## 10.2 The MTP Layer Structure: `NemotronHMTP` and `NemotronHMTPDecoderLayer`

Each MTP prediction head follows the hybrid pattern defined by `mtp_hybrid_override_pattern` in the config, which specifies the interleaving of Mamba and attention blocks within a single MTP step. The [`NemotronHMTP`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_nemotron_h.py) module builds pattern-based layers and attaches a shared prediction head:

```python
class NemotronHMTP(nn.Module):
    def __init__(self, model_config, layer_idx, aux_stream_dict, ...):
        super().__init__()
        self.pattern_str = config.mtp_hybrid_override_pattern
        self.layers = nn.ModuleDict()

        for step_rel_idx in range(self.pattern_len):
            char = self.pattern_str[step_rel_idx]
            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            # MTP layers are always unquantized (NVFP4 checkpoint stores them in bf16)
            sublayer_quant_config = self._get_mtp_sublayer_quant_config(...)
            sublayer_model_config = replace(model_config,
                                            quant_config=sublayer_quant_config,
                                            spec_config=None)

            self.layers[str(step_rel_idx)] = NemotronHMTPDecoderLayer(
                model_config=sublayer_model_config,
                layer_idx=self.layer_idx,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
                layer_type=char,  # 'M' for Mamba, 'A' for Attention, etc.
            )
        # Reuse DeepseekV3's MTP head for the final token prediction
        self.shared_head = DeepseekV3MTPHead(model_config)
```

A key design detail is the **start projection**: the first sublayer in each MTP step fuses the embedding of the previously predicted token with the backbone's hidden state via `enorm`, `hnorm`, and a linear projection `eh_proj`:

```python
class NemotronHMTPDecoderLayer(NemotronHLayer):
    def forward(self, inputs_embeds, hidden_states, residual, attn_metadata, **kwargs):
        if self.has_start_projections:
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)
            fused = torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)
            if mapping.tp_size > 1:
                fused = torch.chunk(fused, mapping.tp_size, dim=-1)[mapping.tp_rank]
            hidden_states = self.eh_proj(fused)
            residual = None  # Start fresh after fusion

        # Proceed through Mamba/Attention mixer...
        hidden_states = self.mixer(hidden_states=hidden_states, attn_metadata=attn_metadata, **kwargs)

        if self.has_end_norm:
            hidden_states, residual = self.final_layernorm(hidden_states, residual)
        return hidden_states, residual
```

The MTP layers in the NVFP4 checkpoint are stored unquantized (bf16). Because TRTLLM's MoE backend only supports FP8/FP4 quantization, the `_get_mtp_sublayer_quant_config` method explicitly overrides `quant_algo=None` for MTP sublayers while preserving the KV cache quantization settings.

## 10.3 The Hard Part: Stateful SSM Caching for Speculative Decoding

In a standard transformer, speculative decoding is relatively straightforward: you extend the KV cache with draft tokens, verify, and truncate the rejected entries. But Nemotron v3 Super's Mamba-2 layers maintain **two types of mutable state** that evolve with every token:

1. **SSM temporal state** `(batch, nheads, dim, dstate)` — the recurrent hidden state from the Selective State Update.
2. **Conv window state** `(batch, conv_dim, d_conv)` — the sliding window for the causal 1D convolution.

During speculative decoding, the model generates `k` draft tokens. If only the first `j < k` are accepted, we need to **roll back** the SSM and conv states to their values after processing the `j`-th token. But the recurrence is sequential — we cannot simply "undo" state updates without replaying from scratch.

The solution is **intermediate state caching**. We introduced a `SpeculativeState` container that extends the base `State` with buffers for every draft step:

```python
@dataclass(frozen=True, kw_only=True)
class State:
    """Base state container for Mamba cache."""
    conv: torch.Tensor     # shape: (num_layers, max_batch, conv_dim, d_conv)
    temporal: torch.Tensor  # shape: (num_layers, max_batch, nheads, dim, dstate)

@dataclass(frozen=True, kw_only=True)
class SpeculativeState(State):
    """Speculative state with intermediate states for draft tokens."""
    intermediate_ssm: torch.Tensor       # (num_layers, max_batch, draft_len, nheads, dim, dstate)
    intermediate_conv_window: torch.Tensor  # (num_layers, max_batch, draft_len, conv_dim, d_conv)
```

The cache manager's `mamba_layer_cache()` method provides per-layer access to all four tensors through a single call, avoiding the overhead of multiple tensor lookups:

```python
def mamba_layer_cache(self, layer_idx: int):
    """Get the Mamba cache state for a specific layer."""
    return self.state.at_layer_idx(layer_idx)
```

## 10.4 Multi-Token State Update Kernel

The Triton SSU kernel (Section 6) was extended to process multiple time steps (the `T` dimension) in a single launch, with intermediate state caching at every step:

```python
# In the Triton kernel: iterate over T draft tokens per sequence
current_step_idx = 0
for _ in range(T):
    # EAGLE-tree custom attention: retrieve parent state if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        if current_step_idx != 0 and cache_idx >= 0:
            parent_step_idx = tl.load(retrieve_parent_token_ptr + ...)
            if parent_step_idx >= 0 and parent_step_idx < T:
                # Load state from the intermediate buffer at the parent's step
                cache_ptr = (intermediate_states_buffer
                             + cache_idx * cache_steps * nheads * dim * dstate
                             + parent_step_idx * nheads * dim * dstate + ...)
                state = tl.load(cache_ptr, mask=mask, other=0.0).to(tl.float32)

    # Standard SSU math: state = state * dA + dB * x
    dA = tl.exp(A * dt[:, None])
    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]

    # Cache intermediate state for this draft step
    if CACHE_INTERMEDIATE_STATES:
        if state_batch_idx != pad_slot_id:
            cache_ptr_base = (intermediate_states_buffer
                              + cache_idx * cache_steps * nheads * dim * dstate
                              + current_step_idx * nheads * dim * dstate + ...)
            tl.store(cache_ptrs, state.to(cache_ptrs.dtype.element_ty), mask=mask)

    out = tl.sum(state * C[None, :], axis=1)
    # ... apply D and z gating, write output ...

    current_step_idx += 1
    # Advance pointers for next time step
    x_ptr += stride_x_T; dt_ptr += stride_dt_T; B_ptr += stride_B_T; ...

# Crucially: disable_state_update=True during verification
# so we DON'T corrupt the main state with speculative draft states
if not DISABLE_STATE_UPDATE:
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)
```

The critical flag is `DISABLE_STATE_UPDATE`. During the target model's verification pass, the main SSM state must **not** be overwritten by the draft token states — the intermediate buffer holds per-step snapshots, and only the accepted steps are later promoted to the main cache.

## 10.5 Triton Causal Conv1d for Speculative Paths

The standard `causal_conv1d_update` processes one token at a time. For MTP, we need to process `draft_token_num` tokens per sequence while maintaining intermediate conv window states for rollback. The PR introduced a new Triton-based [`causal_conv1d_update`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py) that supports multi-token batched updates with intermediate state caching:

```python
# In Mamba2Mixer.forward() — speculative decode path:
if is_target_verify:
    draft_token_num = spec_metadata.max_draft_len + 1
    intermediate_conv_states = layer_cache.intermediate_conv_window

    # Reshape: (num_decode_tokens, conv_dim) -> (num_decodes, conv_dim, draft_token_num)
    xbc_d_reshaped = xbc_d.view(num_decodes, draft_token_num, -1).transpose(1, 2)

    xbc_d_processed = causal_conv1d_update_triton(
        xbc_d_reshaped,
        conv_states,
        self.conv1d.weight,
        self.conv1d.bias,
        activation="silu",
        conv_state_indices=state_indices_d[:num_decodes],
        intermediate_conv_window=intermediate_conv_states,
        intermediate_state_indices=self.intermediate_state_indices,
    )
```

The Triton conv kernel processes all draft tokens in a single launch, updating the conv window state token-by-token and saving intermediate snapshots at each step. This avoids the overhead of `draft_token_num` separate kernel launches while keeping the conv states recoverable.

## 10.6 Cache Promotion: Accepting Verified Draft Tokens

After the target model verifies the draft tokens, the cache manager's `update_resources` method promotes the intermediate states of the accepted tokens into the main cache:

```python
def update_resources(self, scheduled_batch, attn_metadata=None, 
                     num_accepted_draft_tokens=None, ...):
    if not self._is_speculative:
        return
    
    for req in scheduled_batch.generation_requests:
        req_id = req.py_request_id
        slot = self._slot_mapping[req_id]
        num_accepted = req.num_accepted_tokens  # How many draft tokens passed verification
        
        if num_accepted > 0:
            for layer_idx in range(self._num_layers):
                layer_state = self.state.at_layer_idx(layer_idx)
                # Copy the accepted step's intermediate SSM state -> main temporal state
                layer_state.temporal[slot] = layer_state.intermediate_ssm[slot, num_accepted - 1]
                # Copy the accepted step's intermediate conv window -> main conv state
                layer_state.conv[slot] = layer_state.intermediate_conv_window[slot, num_accepted - 1]
```

This promotion step is the key to correctness: if 3 out of 5 draft tokens are accepted, the main SSM and conv states are updated to reflect exactly the state after processing the 3rd accepted token, discarding the speculative states for tokens 4 and 5.

## 10.7 The Complete Speculative Decode Flow

Putting it all together, the speculative decode flow for a single Mamba layer looks like this:

1. **Draft phase**: The MTP layers generate `k` draft tokens. Each token's SSM state update is cached in `intermediate_ssm[step]` and each conv window is cached in `intermediate_conv_window[step]`. The main SSM/conv states are **not** modified (`disable_state_update=True`).

2. **Verify phase**: The target model runs a single forward pass over all `k` draft tokens simultaneously. The multi-token SSU kernel processes all tokens in one launch, using the intermediate state buffer to chain the sequential recurrence correctly (token `i+1` reads from the state produced by token `i`).

3. **Accept phase**: The runtime determines that the first `j` tokens match. `update_resources` copies `intermediate_ssm[:, j-1]` and `intermediate_conv_window[:, j-1]` into the main cache. The next decode step starts with the correct state.

4. **Reject recovery**: Tokens `j+1` through `k` are silently discarded — their intermediate states are simply overwritten in the next speculative cycle. No explicit "rollback" is needed because the main state was never corrupted.

This design achieves speculative decoding for a stateful hybrid Mamba-Attention model with zero wasted memory bandwidth on rejected tokens, while the one-engine approach avoids the latency and memory overhead of maintaining a separate draft model.


# 11. Key Takeaways and Best Practices

Throughout the optimization process of Nemotron v3 Super, I gathered several key takeaways and best practices that might be helpful for others working on LLM inference optimization:

* **Use Proxy Workloads for Fast Iteration:** Running a full performance sweep across all possible batch sizes and sequence lengths is incredibly time-consuming. To iterate quickly, use a proxy workload—such as Input Sequence Length (ISL) = 1K and Output Sequence Length (OSL) = 1K. This provides a fast, directional sense of whether an optimization is working before committing to a full sweep.
* **Leverage AI for Profile Analysis:** A great way to quantify the impact of an optimization is to run Nsight Systems (nsys) profiles before and after the change. You can then feed the raw kernel execution logs into an AI assistant to automatically compare the diffs and calculate the exact speedup ratios, saving hours of manual spreadsheet work.
* **Beware of CUDA Corner Cases:** When implementing custom CUDA kernels, it is easy to focus only on the "happy path" (the standard user case). However, real-world inference involves many corner cases: unaligned memory addresses, weird tensor strides, and unexpected layouts. Always add strict memory guards and fallback handling to prevent your custom kernels from crashing or hanging the GPU. (See [this commit](https://github.com/NVIDIA/TensorRT-LLM/pull/11273/changes/66f64d0a9ba3ec19362fb220238dd3ebbfac8568) for an example of handling unaligned memory).
* **Watch Out for Integer Overflow in Triton:** While Triton abstracts away many of the low-level CUDA memory padding headaches, it introduces its own quirks. One major issue I encountered was integer overflow. When dealing with very large input sequences combined with large batch sizes, the total number of elements can easily exceed `INT32_MAX` (2.14 billion). If your Triton kernel calculates memory offsets using 32-bit integers, it will silently overflow and read/write garbage data. Always cast element indices to `INT64` when calculating pointers for large LLM workloads. (See [this fix](https://github.com/NVIDIA/TensorRT-LLM/pull/12194/changes#diff-5b2be53600df6ffe0ef82b351ae968a7389599f475df038fa92f82e78ecedfe7R45)).


