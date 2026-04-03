---
title: "Deep Dive: AllReduce and AllReduce Fusion in TensorRT-LLM"
date: 2026-04-03
categories: [LLM, optimization]
tags: [LLM]
description: A comprehensive technical guide — from collective communication fundamentals to fused kernel internals, with Nemotron-H as a running example.
---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Collective Communication Fundamentals](#2-background-collective-communication-fundamentals)
   - 2.1 [The Four Core Collective Primitives](#21-the-four-core-collective-primitives)
     - 2.1.1 [AllReduce](#211-allreduce)
     - 2.1.2 [AllGather](#212-allgather)
     - 2.1.3 [ReduceScatter](#213-reducescatter)
     - 2.1.4 [All-to-All](#214-all-to-all)
   - 2.2 [Comparing the Four Collectives](#22-comparing-the-four-collectives)
   - 2.3 [Where Each Collective Appears in an LLM Forward Pass](#23-where-each-collective-appears-in-an-llm-forward-pass)
   - 2.4 [Classic AllReduce Algorithms](#24-classic-allreduce-algorithms)
   - 2.5 [NCCL: NVIDIA's Collective Communication Library](#25-nccl-nvidias-collective-communication-library)
   - 2.6 [GPU Interconnect: NVLink and NVSwitch](#26-gpu-interconnect-nvlink-and-nvswitch)
   - 2.7 [Tensor Parallelism and Where AllReduce Fits](#27-tensor-parallelism-and-where-allreduce-fits)
3. [Communication Optimization Opportunities in Multi-GPU LLM Inference](#3-communication-optimization-opportunities-in-multi-gpu-llm-inference)
   - 3.1 [Intra-Node: Multiple GPUs Connected via NVLink/NVSwitch](#31-intra-node-multiple-gpus-connected-via-nvlinknvswitch)
   - 3.2 [Inter-Node: GPUs Spanning Multiple Servers](#32-inter-node-gpus-spanning-multiple-servers)
   - 3.3 [Cross-Cutting Optimizations (Both Intra- and Inter-Node)](#33-cross-cutting-optimizations-both-intra--and-inter-node)
   - 3.4 [Summary: Optimization Landscape](#34-summary-optimization-landscape)
4. [Meet the Model: Nemotron-H](#4-meet-the-model-nemotron-h)
5. [AllReduce Strategies in TensorRT-LLM](#5-allreduce-strategies-in-tensorrt-llm)
   - 5.1 [Strategy Overview](#51-strategy-overview)
   - 5.2 [Custom IPC Kernels: ONESHOT and TWOSHOT](#52-custom-ipc-kernels-oneshot-and-twoshot)
   - 5.3 [NCCL-Based Strategies: NCCL, NCCL_SYMMETRIC, SYMM_MEM](#53-nccl-based-strategies-nccl-nccl_symmetric-symm_mem)
   - 5.4 [Multi-Node NVLink: MNNVL](#54-multi-node-nvlink-mnnvl)
   - 5.5 [User Buffers (UB): Zero-Copy GEMM-to-AllReduce](#55-user-buffers-ub-zero-copy-gemm-to-allreduce)
   - 5.6 [Strategy Comparison Summary](#56-strategy-comparison-summary)
6. [The Core Optimization: Fusing AllReduce with RMSNorm](#6-the-core-optimization-fusing-allreduce-with-rmsnorm)
   - 6.1 [Why Fusion Matters: The Memory Bandwidth Argument](#61-why-fusion-matters-the-memory-bandwidth-argument)
   - 6.2 [How It Works in Nemotron-H](#62-how-it-works-in-nemotron-h)
   - 6.3 [Inside the Fused Kernel](#63-inside-the-fused-kernel)
   - 6.4 [Memory Traffic Analysis](#64-memory-traffic-analysis)
   - 6.5 [NVFP4 Quantization Fusion](#65-nvfp4-quantization-fusion)
7. [Strategy Selection: From AUTO to the Right Kernel](#7-strategy-selection-from-auto-to-the-right-kernel)
   - 7.1 [The Two-Layer Decision Process](#71-the-two-layer-decision-process)
   - 7.2 [The Static Lookup Table](#72-the-static-lookup-table)
   - 7.3 [The AutoTuner](#73-the-autotuner)
   - 7.4 [Traced Decision Examples](#74-traced-decision-examples)
8. [Platform Topology: How Hardware Shapes Strategy](#8-platform-topology-how-hardware-shapes-strategy)
   - 8.1 [P2P vs NVLink vs MNNVL](#81-p2p-vs-nvlink-vs-mnnvl)
   - 8.2 [Topology Summary: B200 DGX vs NVL72](#82-topology-summary-b200-dgx-vs-nvl72)
9. [Profiling Results: Measured Behavior on B200](#9-profiling-results-measured-behavior-on-b200)
   - 9.1 [Decode Phase: ONESHOT Fusion Confirmed](#91-decode-phase-oneshot-fusion-confirmed)
   - 9.2 [Prefill Phase: When the AutoTuner Chooses Differently](#92-prefill-phase-when-the-autotuner-chooses-differently)
   - 9.3 [Decoding the NCCL Kernel Name](#93-decoding-the-nccl-kernel-name)
10. [Gaps, Trade-offs, and Future Directions](#10-gaps-trade-offs-and-future-directions)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Introduction

In large-scale LLM inference, the forward pass of a model is not a purely compute-bound problem. When a model's parameters are distributed across multiple GPUs via tensor parallelism (TP), every layer must synchronize partial results across devices. This synchronization — the **AllReduce** collective — becomes one of the most critical operations on the inference critical path.

For a model like **Nemotron-H Ultra** with 108 layers running on 4 or 8 GPUs, the forward pass executes 108 AllReduce operations *per token generation step*. Each AllReduce is followed by a residual addition and an RMS normalization. In the unfused case, these are 3 separate CUDA kernels per layer, each reading from and writing to GPU HBM. That is 324 kernel launches, 648 HBM round-trips — per step, per batch.

The key insight behind **AllReduce fusion** is that AllReduce, residual addition, and RMSNorm are all memory-bandwidth-bound operations. By fusing them into a single CUDA kernel that keeps intermediate data in GPU registers, we can eliminate redundant HBM traffic and kernel launch overhead. On NVIDIA Blackwell GPUs (B200), this optimization delivers measurable latency improvements in the decode phase — where these memory-bound operations dominate the execution time.

This blog post is a comprehensive technical deep dive into:

- **How AllReduce works** — from textbook algorithms to the custom GPU kernels in TensorRT-LLM
- **The full strategy landscape** — ONESHOT, TWOSHOT, NCCL, NCCL_SYMMETRIC, SYMM_MEM, MNNVL, and User Buffers
- **How the fused kernel works internally** — C++ template metaprogramming, Lamport synchronization, and compile-time pattern dispatch
- **How the right strategy is selected at runtime** — the two-layer Python/C++ decision process, static lookup tables, and the AutoTuner
- **Real profiling results** on B200 hardware with Nemotron-H

We use NVIDIA's [Nemotron-H](https://research.nvidia.com/labs/adlr/nemotronh/) as a running example throughout — a hybrid Mamba-Transformer-MoE model whose diverse layer types make it an excellent testbed for AllReduce fusion across different computational patterns.

---

## 2. Background: Collective Communication Fundamentals

Distributed LLM inference relies on a small set of **collective communication primitives** — operations where all participating GPUs (ranks) cooperate to exchange or combine data. Understanding these primitives is essential because each one serves a distinct purpose in the parallelism strategy, and choosing the wrong one (or implementing it poorly) can dominate inference latency.

This section covers the four core collectives — AllReduce, AllGather, ReduceScatter, and All-to-All — with their semantics, data movement patterns, and concrete use cases in LLM inference. We then compare them side-by-side before diving into algorithms and hardware.

### 2.1 The Four Core Collective Primitives

#### 2.1.1 AllReduce

AllReduce computes an element-wise reduction (typically summation) across all `N` ranks and distributes the **identical, complete** result to every rank.

```
Before AllReduce (N=4, sum):
  Rank 0: [a0, b0, c0, d0]        After:
  Rank 1: [a1, b1, c1, d1]    →   All ranks: [Σa, Σb, Σc, Σd]
  Rank 2: [a2, b2, c2, d2]        (identical full result everywhere)
  Rank 3: [a3, b3, c3, d3]

Input size per rank:  D
Output size per rank: D  (same shape, fully reduced)
```

**Semantics:** Every rank provides an input tensor of the same shape. AllReduce applies a binary operator (sum, min, max) element-wise across all inputs and replicates the result to every rank. Each rank ends up with an identical copy of the fully-reduced tensor.

**LLM inference use case — Row-parallel linear layers (Tensor Parallelism):** In the Megatron-LM TP pattern [[Shoeybi et al., 2019]](#12-references), row-parallel layers (e.g., `o_proj` after attention, `down_proj` after MLP) produce partial sums — each rank computes `output = input_shard @ weight_shard`, resulting in the same-shaped output with different values. AllReduce sums these partial results so every rank has the correct, full output.

In TensorRT-LLM, this is implemented in the `Linear` module (`tensorrt_llm/_torch/modules/linear.py`):

```python
# Row-parallel: each rank has a partial sum → AllReduce to get the full result
if self.reduce_output:
    output = self.apply_linear(input, bias, lora_params, layer_idx)
    output = self.all_reduce(output, all_reduce_params=all_reduce_params)
```

The C++ dispatch (`allreduceOp.cpp`) routes to one of several implementations — custom ONESHOT/TWOSHOT kernels, NCCL, or NCCL_SYMMETRIC — depending on message size and hardware topology (detailed in [Section 5](#5-allreduce-strategies-in-tensorrt-llm)).

**When to use:** Whenever every rank needs the full, reduced result and the next computation consumes the entire tensor. This is the most common collective in TP-based LLM inference.

#### 2.1.2 AllGather

AllGather collects data from all `N` ranks and **concatenates** them along a specified dimension. No reduction (summation) occurs — each rank's contribution is preserved intact in the output.

```
Before AllGather (N=4, along dim=0):
  Rank 0: [a0, b0]                After (all ranks):
  Rank 1: [a1, b1]            →   [a0, b0, a1, b1, a2, b2, a3, b3]
  Rank 2: [a2, b2]                (concatenation of all inputs)
  Rank 3: [a3, b3]

Input size per rank:  D/N  (each rank holds a shard)
Output size per rank: D    (full concatenated tensor)
```

**Semantics:** Each rank contributes its local tensor. AllGather concatenates all contributions in rank order along the gather dimension. Every rank receives the identical, concatenated result. No mathematical operation is applied — it is purely a data assembly operation.

In TensorRT-LLM (`tensorrt_llm/_torch/distributed/ops.py`):

```python
def allgather(input, mapping, dim=-1, sizes=None):
    '''
    Performs a collective all-gather across the TP group.
    Input tensors are concatenated at dimension 'dim' to produce the output.
    output.shape[dim] = input.shape[dim] * tp_group_size
    
    Implemented via NCCL all-gather or grouped NCCL broadcast.
    '''
```

The C++ implementation (`allgatherOp.cpp`) calls `ncclAllGather` for uniform-sized inputs, or falls back to per-root `ncclBroadcast` for variable-sized shards.

**LLM inference use cases:**

1. **Column-parallel linear layers (TP):** Column-parallel layers (e.g., `qkv_proj`, `gate_up_proj`) shard the output dimension. Each rank produces `[tokens, out_features/TP]`. When subsequent computation needs the full width (e.g., the embedding lookup or final logits), AllGather reassembles the shards:

```python
# Column-parallel: each rank has an output shard → AllGather to reassemble
elif self.tp_mode == TensorParallelMode.COLUMN:
    output = self.apply_linear(input, self.bias, lora_params, layer_idx)
    if self.gather_output:
        output = allgather(output, self.mapping)
```

2. **MoE dispatch phase (Expert Parallelism):** In Mixture-of-Experts with attention-DP, each rank holds a subset of tokens. Before running experts, AllGather collects all tokens from all ranks so every rank has the full token set for local expert computation:

```python
# MoE dispatch: gather tokens from all DP ranks before expert computation
class AllGatherReduceScatter(Communication):
    def dispatch(self, hidden_states, ...):
        hidden_states, ... = allgather(
            [hidden_states, ...], self.mapping, dim=0, sizes=sizes)
```

3. **Context Parallelism (Helix):** When sequence chunks are distributed across CP ranks, AllGather reassembles the full sequence before operations that need global context (e.g., the LM head).

**When to use:** Whenever each rank holds a **shard** of a tensor (split along some dimension) and you need the **full, unsplit** tensor on every rank. No reduction — just concatenation.

#### 2.1.3 ReduceScatter

ReduceScatter performs an element-wise reduction (like AllReduce) but then **partitions** the result, giving each rank only its `1/N` shard. It can be thought of as AllReduce followed by a scatter, or equivalently as the inverse of AllGather with a reduction step.

```
Before ReduceScatter (N=4, sum, partition along dim=0):
  Rank 0: [a0, b0, c0, d0, e0, f0, g0, h0]
  Rank 1: [a1, b1, c1, d1, e1, f1, g1, h1]
  Rank 2: [a2, b2, c2, d2, e2, f2, g2, h2]
  Rank 3: [a3, b3, c3, d3, e3, f3, g3, h3]

After ReduceScatter:
  Rank 0: [Σa, Σb]                 (1/4 of the reduced result)
  Rank 1: [Σc, Σd]                 (next 1/4)
  Rank 2: [Σe, Σf]                 (next 1/4)
  Rank 3: [Σg, Σh]                 (last 1/4)

Input size per rank:  D
Output size per rank: D/N  (only this rank's shard of the reduced result)
```

**Semantics:** All ranks provide same-shaped inputs. The inputs are reduced element-wise (e.g., summed), then the result is split into `N` equal chunks, with rank `i` receiving chunk `i`. Each rank ends up with a different, non-overlapping portion of the fully-reduced tensor.

In TensorRT-LLM (`tensorrt_llm/_torch/distributed/ops.py`):

```python
def reducescatter(input, mapping, dim=-1, sizes=None):
    # Calls torch.ops.trtllm.reducescatter → ncclReduceScatter (uniform split)
    # or multiple ncclReduce calls (variable split)
```

**LLM inference use cases:**

1. **MoE combine phase (Expert Parallelism):** After AllGather-based dispatch and local expert computation, each rank has computed expert outputs for *all* tokens (duplicated across ranks). ReduceScatter sums the contributions and splits the result so each rank gets back only its local token shard — the inverse of the dispatch AllGather:

```python
# MoE combine: reduce expert outputs and scatter back to per-rank token shards
class AllGatherReduceScatter(Communication):
    def combine(self, final_hidden_states, **kwargs):
        outputs = reducescatter(
            final_hidden_states, self.mapping, dim=0,
            sizes=self._dispatch_state.get("sizes"))
        return outputs
```

2. **Helix CP attention output:** After the attention output projection, ReduceScatter sums partial sums across the CP group and scatters the result so each CP rank processes a distinct token chunk through the subsequent MLP:

```python
# Helix CP: reduce-scatter after o_proj so each CP rank gets its token chunk for MLP
def _helix_cp_output_projection(o_proj, attn_output, ...):
    attn_output = o_proj(attn_output, all_reduce_params=AllReduceParams(enable_allreduce=False))
    attn_output = reducescatter(attn_output, mapping_o, dim=0)
```

**When to use:** When the next computation only needs a **shard** of the reduced result (not the full tensor). This saves memory bandwidth compared to AllReduce — each rank writes and stores only `D/N` instead of `D`. ReduceScatter is the natural partner of AllGather: the pair `AllGather → compute → ReduceScatter` is equivalent to `compute → AllReduce` but allows intermediate computation on the full, gathered tensor.

#### 2.1.4 All-to-All

All-to-All is the most general collective. Each rank sends a **different** chunk of data to each peer rank, and receives a different chunk from each peer. It is a full data permutation — not a reduction, not a concatenation, but a **redistribution** of data across ranks.

```
Before All-to-All (N=4):
  Rank 0: [a→0, a→1, a→2, a→3]     (4 chunks, one destined for each rank)
  Rank 1: [b→0, b→1, b→2, b→3]
  Rank 2: [c→0, c→1, c→2, c→3]
  Rank 3: [d→0, d→1, d→2, d→3]

After All-to-All:
  Rank 0: [a→0, b→0, c→0, d→0]     (received one chunk from each rank)
  Rank 1: [a→1, b→1, c→1, d→1]
  Rank 2: [a→2, b→2, c→2, d→2]
  Rank 3: [a→3, b→3, c→3, d→3]

Input size per rank:  D  (N chunks of D/N)
Output size per rank: D  (N chunks of D/N, from different sources)
```

**Semantics:** Each rank provides `N` data chunks (one per destination rank) and receives `N` chunks (one from each source rank). Data is neither summed nor concatenated — it is **routed**. Rank `i` sends its `j`-th chunk to rank `j`, and receives rank `j`'s `i`-th chunk.

TensorRT-LLM implements several All-to-All variants for different use cases:

```python
# Ulysses-style sequence parallelism: switch between seq-sharded and head-sharded layouts
def all_to_all_4d(input, scatter_dim, gather_dim, process_group):
    """
    Redistributes a 4D tensor [batch, seq, heads, head_dim] using all-to-all.
    - Sequence sharding [B, S/P, H, D] → Head sharding [B, S, H/P, D]
    - Head sharding [B, S, H/P, D] → Sequence sharding [B, S/P, H, D]
    """

# Helix CP: exchange partial attention outputs and softmax stats across CP ranks
def alltoall_helix(inputs, group):
    """All-to-all across a given group using NCCL send/recv operations."""

# MoE Expert Parallelism: route tokens to the experts that own them
class MoeAlltoAll:
    """MoE all-to-all using NVLink one-sided, two-sided, or DeepEP backends."""
```

**LLM inference use cases:**

1. **MoE Expert Parallelism — Token Routing:** This is the primary use case for All-to-All in LLM inference. When experts are distributed across ranks (Expert Parallelism, EP), each rank's tokens may need to reach experts on *different* ranks. All-to-All routes each token to the rank that owns its assigned expert, and routes the results back afterward. The MoE communication factory in TensorRT-LLM selects among multiple All-to-All implementations:

```python
# Communication strategy selection for MoE (communication_factory.py):
# Selection priority:
# 1. NVLinkOneSided  (highest priority for throughput)
# 2. NVLinkTwoSided  (high priority for latency)
# 3. DeepEP          (if enabled)
# 4. DeepEPLowLatency
# 5. AllGather + ReduceScatter  (fallback, always works)
```

   When `enable_alltoall=True`, the system uses direct All-to-All token routing instead of the AllGather + ReduceScatter fallback. This is more bandwidth-efficient because each token is sent only to the one rank that needs it, rather than being broadcast to all ranks.

2. **Ulysses Sequence Parallelism — Layout Transformation:** In Ulysses-style sequence parallelism, the attention computation is distributed across ranks by splitting the sequence dimension. Before attention, each rank holds `[B, S/P, H, D]` (a sequence shard with all heads). All-to-All transforms this to `[B, S, H/P, D]` (the full sequence but only some heads) so each rank can compute attention for its head subset over the full sequence. After attention, a reverse All-to-All transforms back.

3. **Helix CP — Partial Attention Exchange:** In the Helix context-parallelism scheme, All-to-All exchanges partial attention outputs and softmax normalization statistics across CP ranks to reconstruct the correct attention result from independently computed chunks.

**When to use:** When each rank needs to send **different data to different destinations** — a data permutation rather than a reduction or broadcast. The canonical case is MoE expert parallelism where tokens are routed to the ranks that own their assigned experts.

### 2.2 Comparing the Four Collectives

The following table summarizes the key differences:

| Property | **AllReduce** | **AllGather** | **ReduceScatter** | **All-to-All** |
|----------|:------------:|:------------:|:-----------------:|:-------------:|
| **Operation** | Reduce + Replicate | Concatenate | Reduce + Partition | Permute/Route |
| **Math applied** | Sum (or min/max) | None | Sum (or min/max) | None |
| **Input per rank** | `D` | `D/N` | `D` | `D` (N chunks) |
| **Output per rank** | `D` (full, identical) | `D` (full, identical) | `D/N` (shard) | `D` (N chunks, from different sources) |
| **Every rank gets same result?** | Yes | Yes | **No** (each gets its shard) | **No** (each gets different data) |
| **Total data moved** | `~2D` (bandwidth-optimal) | `D × (N-1)/N` | `D × (N-1)/N` | `D × (N-1)/N` |
| **Primary LLM use** | TP row-parallel sync | TP column-parallel gather, MoE dispatch | MoE combine, Helix CP | MoE EP routing, Ulysses SP |

**Relationships between them:**
- **AllReduce = ReduceScatter + AllGather.** In fact, this is how ring AllReduce is implemented internally. The NCCL library can decompose AllReduce into these two phases for pipelined execution.
- **AllGather + ReduceScatter** is the MoE fallback for All-to-All. When true All-to-All is unavailable, MoE dispatch uses AllGather (broadcast all tokens to all ranks), and MoE combine uses ReduceScatter (sum expert outputs and scatter back). This works but sends more data than necessary — every token goes to every rank even if only one rank's expert is needed.

### 2.3 Where Each Collective Appears in an LLM Forward Pass

To make this concrete, here is where each collective appears during a single forward pass of a tensor-parallel LLM with MoE layers:

```
Embedding Lookup
  ├── Column-parallel embedding → AllGather (reassemble full vocabulary shard)
  │
  ▼
Transformer / Hybrid Layer (repeated N times)
  │
  ├── Attention
  │   ├── qkv_proj (column-parallel) — no communication
  │   ├── attention computation
  │   │   └── [Ulysses SP: All-to-All before attention, All-to-All after]
  │   │   └── [Helix CP: All-to-All to exchange partial attention stats]
  │   └── o_proj (row-parallel) → AllReduce (sum partial sums across TP ranks)
  │       └── [Helix CP+DP: ReduceScatter instead of AllReduce]
  │
  ├── Residual Add + RMSNorm
  │   └── [With fusion: folded into the AllReduce kernel above]
  │
  ├── FFN / MoE
  │   ├── Dense MLP:
  │   │   ├── gate_up_proj (column-parallel) — no communication
  │   │   └── down_proj (row-parallel) → AllReduce
  │   │
  │   └── MoE (Expert Parallel):
  │       ├── Router → dispatch:
  │       │   └── All-to-All (route tokens to expert owners)
  │       │   └── [Fallback: AllGather (broadcast all tokens)]
  │       ├── Expert computation (local)
  │       └── Combine:
  │           └── All-to-All (route results back)
  │           └── [Fallback: ReduceScatter (sum + shard)]
  │       └── Optional: AllReduce (finalize combined expert output)
  │
  ▼
Final Norm → LM Head
  └── [CP: AllGather to reassemble full sequence for logits]
```

### 2.4 Classic AllReduce Algorithms

Since AllReduce is the most performance-critical collective in TP-based inference (and the focus of this blog's fusion optimization), we examine its internal algorithms in more detail.

AllReduce can be decomposed into two phases: **ReduceScatter** (each rank ends up with a fully-reduced `1/N` chunk) followed by **AllGather** (each rank broadcasts its chunk to all others).

#### Ring AllReduce

The ring algorithm arranges ranks in a logical ring. In the ReduceScatter phase, each rank sends a chunk to its neighbor and receives + accumulates a chunk from the other side, rotating `N-1` times. The AllGather phase similarly rotates fully-reduced chunks around the ring.

```
Ring AllReduce (N=4):
  Phase 1 — Reduce-Scatter: N-1 = 3 steps
    Step 1: Rank 0 → Rank 1, Rank 1 → Rank 2, Rank 2 → Rank 3, Rank 3 → Rank 0
    Step 2: rotated chunks accumulate...
    Step 3: each rank has 1/4 of the fully-reduced data

  Phase 2 — All-Gather: N-1 = 3 steps
    Fully-reduced chunks rotate around the ring until all ranks have the full result

  Total: 2×(N-1) communication steps
  Data per step: D/N  (D = total data size)
  Total traffic per rank: 2 × (N-1)/N × D ≈ 2D  (bandwidth-optimal)
```

The ring algorithm is **bandwidth-optimal** — each rank sends and receives the minimum total data. However, it requires `2×(N-1)` sequential steps, making it **latency-suboptimal** for small messages where per-step overhead dominates [[Patarasuk & Yuan, 2009]](#12-references).

#### Tree AllReduce

The tree algorithm organizes ranks into a binary tree. It performs a Reduce (bottom-up aggregation to the root) followed by a Broadcast (top-down distribution from the root). This requires only `2×log₂(N)` steps — better latency — but each step involves more data movement. In practice, NCCL combines tree and ring approaches, selecting the better one based on message size [[NCCL Docs]](#12-references).

#### Butterfly (Recursive Halving/Doubling)

The butterfly algorithm uses `log₂(N)` rounds of pairwise exchanges. It is both bandwidth-optimal and latency-optimal for power-of-two process counts. However, it can cause network contention in systems with hierarchical topologies [[Patarasuk & Yuan, 2009]](#12-references). Recent work has shown that optimal non-pipelined reduce-scatter can be achieved in `⌈log₂(N)⌉` communication rounds with a simple circulant graph pattern [[Li et al., 2024]](#12-references).

### 2.5 NCCL: NVIDIA's Collective Communication Library

[NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) is the standard library for multi-GPU and multi-node collective operations. When you call `ncclAllReduce()`, NCCL internally selects between algorithms and protocols based on message size, topology, and GPU generation:

- **Ring algorithm**: Fuses Reduce-Scatter and AllGather into a single pipeline. Better for large messages where bandwidth dominates.
- **Tree algorithm**: Performs Reduce + Broadcast per channel. Better for small messages where latency dominates.
- **Protocols**: NCCL selects between Simple (direct copy), LL (Low Latency, with inline data), and LL128 (128-byte low-latency) protocols [[NCCL GitHub Issues #256, #530]](#12-references).

NCCL manages memory staging, channel allocation, and NIC utilization transparently. It is the universal fallback when custom kernels cannot be used.

### 2.6 GPU Interconnect: NVLink and NVSwitch

The physical interconnect between GPUs fundamentally determines which AllReduce strategies are viable and how fast they can run.

**NVLink** is NVIDIA's high-bandwidth GPU-to-GPU interconnect:
- **NVLink4 (Hopper / H100)**: 900 GB/s bidirectional per GPU
- **NVLink5 (Blackwell / B200)**: 1.8 TB/s bidirectional per GPU — a 2× improvement

**NVSwitch** is a crossbar switch that provides all-to-all NVLink connectivity within a node (and across nodes in NVL72 systems). Rather than point-to-point links, NVSwitch enables any GPU to communicate with any other GPU at full NVLink bandwidth simultaneously.

**MULTIMEM** is a hardware-level instruction available on Hopper+ GPUs with NVSwitch. It enables a single GPU to issue a **multicast load** (LDMC) or **multicast store** (STMC) that the NVSwitch hardware fans out to all peer GPUs in a single operation. This provides O(1) communication complexity per GPU, regardless of the number of ranks — the theoretical ideal for collective operations [[NVIDIA NVLink Blog]](#12-references).

**GB200 NVL72**: NVIDIA's rack-scale system connecting 72 Blackwell GPUs across 18 nodes (4 GPUs per node) via NVSwitch fabric. The NVLink domain spans the entire rack, enabling multi-node GPU-to-GPU communication at NVLink speeds — a capability known as **Multi-Node NVLink (MNNVL)** [[NVIDIA GB200 Blog]](#12-references).

### 2.7 Tensor Parallelism and Where AllReduce Fits

In tensor parallelism (TP), model weight matrices are sharded across GPUs. The standard pattern from Megatron-LM [[Shoeybi et al., 2019]](#12-references) alternates two types of parallelism in each transformer layer:

1. **Column-parallel**: The weight matrix is split along the output dimension. Each rank computes a shard of the output independently — no communication needed.
2. **Row-parallel**: The weight matrix is split along the input dimension. Each rank computes a partial sum of the full output. An **AllReduce** is required to sum these partial results across ranks.

```
Column-parallel (e.g., qkv_proj, gate_up_proj):
  Weight: [hidden, out/TP]  per rank
  Output: [tokens, out/TP]  ← sharded, no communication

Row-parallel (e.g., o_proj, down_proj):
  Weight: [hidden/TP, out]  per rank
  Output: [tokens, out]     ← full-size partial sum, needs AllReduce
```

Crucially, the AllReduce message size is `total_tokens × hidden_size × dtype_size`, independent of `tp_size`. The tensor is **not** sharded by TP for the AllReduce — each rank holds a same-shaped partial sum that differs in values. For Nemotron-H with `hidden_size=8192` and bf16 dtype, each AllReduce moves `total_tokens × 16 KB` of data.

A typical transformer layer has **two** AllReduce points: after the attention output projection and after the MLP down projection. Nemotron-H's hybrid architecture adds variety — Mamba layers, MoE layers, and transformer layers all have row-parallel projections that need AllReduce.

---

## 3. Communication Optimization Opportunities in Multi-GPU LLM Inference

Communication is not just a necessary overhead — it is frequently the **dominant bottleneck** in distributed LLM inference. Research has shown that AllReduce alone can consume 30–50% of end-to-end latency during tensor-parallel inference [[Meta Engineering Blog, 2025]](#12-references), and NVLink bandwidth improvements (3× from A100 to B200) have not kept pace with tensor core throughput improvements (7.2×) [[ParallelKittens, 2025]](#12-references). This growing gap makes communication optimization increasingly critical.

This section provides a systematic analysis of optimization opportunities, organized by the two fundamental deployment topologies: intra-node (multiple GPUs within a single server) and inter-node (GPUs spanning multiple servers). For each, we examine the hardware constraints, the available optimization techniques, and how TensorRT-LLM implements them.

### 3.1 Intra-Node: Multiple GPUs Connected via NVLink/NVSwitch

Within a single DGX or HGX node, GPUs communicate over NVLink through an NVSwitch crossbar. This is the highest-bandwidth, lowest-latency interconnect available — but even here, communication time is significant and multiple optimization opportunities exist.

#### 3.1.1 Optimization 1: Bypass NCCL with Direct IPC Kernels

**The problem:** NCCL is a general-purpose library that works across any topology. For intra-node NVLink communication, this generality comes at a cost: NCCL stages data through internal buffers, uses ring or tree algorithms with multiple steps, and incurs library dispatch overhead.

**The optimization:** TensorRT-LLM's custom ONESHOT and TWOSHOT kernels bypass NCCL entirely, using **CUDA IPC** (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) to map each GPU's memory into every other GPU's address space. CUDA kernels can then directly dereference pointers to remote GPU memory, reading and writing over NVLink without any library intermediary.

The IPC workspace is set up at initialization (`tensorrt_llm/_ipc_utils.py`):

```python
# IPC is only available for intra-node GPUs
class IpcMemory:
    def open_ipc_memory(self):
        # cudaIpcGetMemHandle → share handle via MPI allgather → cudaIpcOpenMemHandle
        # Result: each GPU can directly access every other GPU's buffer
```

This is guarded by a topology check — IPC is disabled if any TP peer is on a different node:

```cpp
// allreduceOp.cpp — inter-node forces P2P off
if (is_inter_node) {
    mIsP2PSupported = false;  // CUDA IPC requires same OS node
}
```

**Measured impact:** Custom IPC kernels achieve 7.9–12.7 µs per AllReduce call in the decode phase on B200 (TP=4, 64 tokens), versus NCCL's typical 20–50 µs for the same message size. The benefit is most pronounced for small messages (≤128 tokens) where per-call latency dominates over bandwidth.

#### 3.1.2 Optimization 2: Kernel Fusion (AllReduce + Residual + Norm + Quant)

**The problem:** Even with fast AllReduce, the subsequent residual addition, RMSNorm, and optional quantization are separate CUDA kernels, each incurring HBM read-write round-trips and launch overhead.

**The optimization:** Fuse all post-AllReduce operations into the AllReduce kernel itself. Data flows from NVLink IPC buffers directly into GPU registers, where residual addition, RMSNorm, and NVFP4 quantization are applied before a single write to HBM. This eliminates 50% of HBM traffic per layer (detailed in [Section 6](#6-the-core-optimization-fusing-allreduce-with-rmsnorm)).

**Implementation:** The fusion is achieved through C++ template metaprogramming — seven fusion patterns are compiled as separate kernel instantiations, selected at compile time via `if constexpr` with zero runtime overhead. The fused operation operator processes data in float4 (128-bit) chunks:

```cpp
// allReduceFusionKernels.cu — fused operator applied to each data element
float4 sum_val = allreduce_sum<DType, NRanks>(vals);   // AllReduce (IPC reads)
fused_op(sum_val, tidx);                                // residual + norm + quant (registers)
```

**Scope:** This optimization applies exclusively to intra-node custom kernels (ONESHOT/TWOSHOT). When NCCL is the AllReduce backend (e.g., for large messages or inter-node), fusion degrades to "pseudo-fusion" — separate `ncclAllReduce` and `residualRmsNorm` kernels.

#### 3.1.3 Optimization 3: NVSwitch Hardware Multicast (LDMC/STMC)

**The problem:** Software-based AllReduce (even custom IPC kernels) requires each GPU to explicitly send data to each peer, resulting in O(N) NVLink traffic per rank for ONESHOT, or O(1) with two explicit software barriers for TWOSHOT.

**The optimization:** NVSwitch on Hopper+ GPUs supports hardware multicast instructions — **LDMC** (Load Multicast) and **STMC** (Store Multicast). A single GPU issues one multicast instruction, and the NVSwitch fabric simultaneously fans out the operation to all peer GPUs. This achieves O(1) communication complexity with **hardware-level ordering** — no software barriers needed.

TensorRT-LLM accesses this through two paths:

1. **NCCL_SYMMETRIC:** NCCL's symmetric kernel (`ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC`) uses LDMC for the reduce-scatter phase and STMC for the allgather phase. Buffers are pre-registered via `ncclMemAlloc` + `ncclCommWindowRegister`:

```cpp
// allreduceOp.cpp — empirically-tuned threshold for when registration is worthwhile
double const a = -4986.43478503;  // linear model fitted on GB200
double const b = 156716.52177552;
size_t minRegistrationThreshold = max(0, a * nRanks + b) * element_size;
```

2. **SYMM_MEM:** PyTorch's symmetric memory API directly issues `multimem_all_reduce_()` — the most direct path to hardware, but limited to `fusion_op=NONE`.

**Trade-off:** Hardware multicast provides the fastest raw AllReduce, but cannot perform post-processing (residual, norm, quant). The AutoTuner resolves this: for decode (small messages), ONESHOT fusion wins; for prefill (large messages), NCCL_SYMMETRIC multicast wins. The crossover is workload-dependent and determined by runtime benchmarking.

#### 3.1.4 Optimization 4: GEMM-AllReduce Fusion (Eliminating the GEMM→AllReduce Copy)

**The problem:** In the standard pipeline, a GEMM (matrix multiply) writes its output to HBM, then the AllReduce kernel reads it back from HBM to push to peers. This GEMM→HBM→AllReduce round-trip is wasteful.

**The optimization:** Fuse the GEMM and AllReduce into a single kernel so the GEMM output flows directly into the communication buffer without an HBM round-trip. TensorRT-LLM implements this at two levels:

1. **CUTLASS-based GEMM+AllReduce** (SM100+, Blackwell): A single Cutlass kernel performs the matrix multiply and writes the output directly to an NVLS (NVLink SHARP) multicast buffer. Implemented in `cpp/tensorrt_llm/kernels/cutlass_kernels/allreduce_gemm/`:

```python
# linear.py — fused GEMM+AllReduce requires SM≥100, NVFP4, row-parallel, NVLink
self.use_fused_gemm_allreduce = all([
    self.reduce_output, dtype_supported,
    in_features_aligned, out_features_aligned,
    device_supported,  # SM ≥ 100
    ipc_nvls_supported(),
])
```

2. **User Buffers (UB)** via torch.compile: Graph pattern matching rewrites `GEMM → AllReduce` sequences so the GEMM writes directly to a pre-registered IPC buffer.

**Constraints:** GEMM+AllReduce fusion requires `TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED=1` (opt-in), SM100+ (Blackwell), NVFP4 quantization, and is incompatible with the cross-layer AllReduce+norm fusion (they target different optimization axes).

#### 3.1.5 Optimization 5: Completion Signal Overlap

**The problem:** After the fused AllReduce+norm kernel completes, a "completion signal" must be written to update the triple-buffer flag for the next iteration. This write is serialized with the kernel's useful work.

**The optimization:** The `trigger_completion_at_end` parameter controls whether the completion signal is written at the end of the kernel (blocking) or deferred. When set to `False` (the default for fused operations), the completion signal write can overlap with the next layer's compute:

```cpp
// allReduceFusionKernels.cu — template parameter controls completion timing
if (trigger_completion_at_end)
    launch_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, true>(params, cfg);
else
    launch_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, false>(params, cfg);
```

Additionally, on SM≥90 (Hopper+), TensorRT-LLM uses **programmatic stream serialization** (`cudaLaunchAttributeProgrammaticStreamSerialization`) for the fused kernels, enabling finer-grained overlap between consecutive kernel launches.

#### 3.1.6 Optimization 6: Low-Precision Communication for PCIe Topologies

**The problem:** Not all multi-GPU systems have NVLink. PCIe-connected GPUs (e.g., some cloud instances) have 10–30× less interconnect bandwidth (PCIe Gen5: ~64 GB/s vs NVLink5: 1.8 TB/s). Standard AllReduce saturates PCIe bandwidth quickly.

**The optimization:** The LOWPRECISION strategy quantizes data to FP8 before P2P communication, then dequantizes after the reduction. This cuts NVLink/PCIe traffic by ~2× at the cost of minor precision loss:

```cpp
// allreduceOp.cpp — LOWPRECISION activates only on PCIe-only topologies
bool isUsingLowPrecision(size_t message_size) const noexcept {
    return force_low_precision && !mIsNVLINKSupported && mIsP2PSupported
        && message_size >= 2 * 1024 * 1024;  // minimum 2 MB
}
```

Dedicated kernels in `customLowPrecisionAllReduceKernels.cu` handle the FP8 quantize → P2P exchange → dequantize → reduce pipeline.

#### 3.1.7 Optimization 7: Heterogeneous Link Aggregation

**The problem:** Even in NVLink-equipped nodes, PCIe and RDMA NIC bandwidth sits idle during intra-node collectives.

**The optimization:** Research has shown that aggregating NVLink, PCIe, and RDMA NICs into a single communication fabric can improve AllReduce bandwidth by 26–27% on 8-GPU H800 systems by offloading 2–22% of traffic to underutilized interconnects [[FlexLink, 2025]](#12-references). While TensorRT-LLM does not currently implement heterogeneous link aggregation natively, this represents a future optimization opportunity.

### 3.2 Inter-Node: GPUs Spanning Multiple Servers

Multi-node inference introduces fundamentally different constraints. Inter-node communication traverses network fabric (InfiniBand, Ethernet, or NVSwitch fabric for NVL72), with higher latency and lower bandwidth than intra-node NVLink. The optimization strategies must account for this two-level hierarchy.

#### 3.2.1 The Topology Detection Foundation

TensorRT-LLM's strategy selection begins with topology detection. The C++ runtime probes the TP group to determine which GPUs share a node and which span nodes:

```cpp
// allreduceOp.cpp — setGroupTopology()
std::set<int> local_group = getLocalGroup(mGroup);  // GPUs on same OS node
bool is_inter_node = (mGroup.size() != local_group.size());

if (is_inter_node) {
    // Probe NVLink within local subgroup
    // Probe MNNVL fabric across nodes
    mIsP2PSupported = false;    // CUDA IPC doesn't work across nodes
    mIsMNNVLSupported = mIsNVLINKSupported && allRanksMnnvlConnected;
}
```

The Python `Mapping` class provides a complementary check:

```python
# mapping.py
def is_multi_node(self):
    return self.world_size > self.gpus_per_node
```

This topology information drives every subsequent optimization decision.

#### 3.2.2 Optimization 1: MNNVL — Multi-Node NVLink AllReduce

**The problem:** On GB200 NVL72 systems, TP groups can span multiple nodes (e.g., TP=8 across 2 nodes of 4 GPUs each). CUDA IPC is unavailable across node boundaries, so the custom ONESHOT/TWOSHOT kernels cannot be used. Falling back to standard NCCL loses the benefits of the NVSwitch fabric connecting the nodes.

**The optimization:** The MNNVL (Multi-Node NVLink) strategy uses `ncclMemAlloc` to allocate multicast-capable memory accessible across the NVSwitch fabric, bypassing the CUDA IPC limitation. The MNNVL kernel implements a Lamport-style one-shot/two-shot protocol over these fabric-managed buffers:

```python
# ops.py — MNNVL activation conditions
@staticmethod
def is_mnnvl():
    return all([
        platform.machine() == "aarch64",          # GB200 Grace CPU
        mapping.is_multi_node(),                    # TP spans nodes
        MnnvlMemory.supports_mnnvl(),              # NVSwitch fabric available
        not mapping.has_cp(),                       # no context parallelism
    ])
```

MNNVL supports true fusion for `RESIDUAL_RMS_NORM` but currently lacks NVFP4/FP8 fusion (an implementation gap, not a hardware limitation — see [Section 10](#10-gaps-trade-offs-and-future-directions)).

**Where it activates:** Exclusively on GB200 NVL72 with TP spanning multiple nodes — the only production configuration where aarch64 + multi-node + NVSwitch fabric conditions are all met.

#### 3.2.3 Optimization 2: Hierarchical Communication

**The problem:** In a flat AllReduce across nodes, every GPU communicates with every other GPU, including across the slow inter-node network. This wastes intra-node bandwidth on data that could be reduced locally first.

**The optimization:** Hierarchical (or multi-level) AllReduce decomposes the operation into intra-node and inter-node phases:

```
Hierarchical AllReduce:
  Phase 1: Intra-node reduce (fast NVLink, e.g., ONESHOT/TWOSHOT)
     Each node produces a single partially-reduced result

  Phase 2: Inter-node AllReduce (slower InfiniBand/NVSwitch fabric)
     One representative per node exchanges data across nodes

  Phase 3: Intra-node broadcast (fast NVLink)
     Result distributed within each node
```

NCCL implements hierarchical decomposition internally when it detects a multi-node topology — the PAT (Parallel Aggregated Trees) algorithm in NCCL 2.23+ provides logarithmic scaling for reduce-scatter and allgather on multi-node configurations [[NCCL 2.23 Blog]](#12-references). TensorRT-LLM does not implement its own hierarchical scheme; it relies on NCCL's internal topology-aware algorithm selection for the multi-node case.

Research beyond NCCL has demonstrated further gains: **NVRAR** (an NVSHMEM-based recursive-doubling AllReduce) achieves 1.9–3.6× lower latency than NCCL for message sizes between 128 KB and 2 MB, translating to 1.72× reduction in end-to-end batch latency for Llama 3.1 405B inference [[NVRAR, 2025]](#12-references).

#### 3.2.4 Optimization 3: Disaggregated Serving — Separating Prefill and Decode

**The problem:** Prefill (processing the input prompt) and decode (generating tokens one at a time) have fundamentally different compute profiles. Prefill is compute-bound (large GEMMs); decode is memory-bandwidth-bound (small per-step compute, KV cache access dominates). Mixing them on the same GPU leads to suboptimal utilization.

**The optimization:** Disaggregated serving separates prefill and decode onto different GPU pools, connected via a high-speed KV cache transfer mechanism. After the prefill node computes the KV cache for a prompt, it transfers the cache to a decode node that handles generation.

TensorRT-LLM implements disaggregated serving with multiple transport backends:

```python
# tensorrt_llm/_torch/disaggregation/native/transfer.py
class TransferWorker:
    # Coordinates KV cache transfer between prefill and decode nodes
    # Backends: NIXL (default), UCX, MPI

# tensorrt_llm/_torch/disaggregation/nixl/agent.py
class NixlTransferAgent:
    # NIXL-based KV cache transfer
    # Supports push (prefill→decode) and pull (decode←prefill) modes
```

**NIXL** (NVIDIA Interconnect eXchange Library) is the default transport, supporting both push-based (prefill proactively sends KV blocks) and pull-based (decode requests KV blocks) transfer modes. Push-based transfer achieves 1.2–3.0× TTFT improvement over pull mode by overlapping transfer with computation [[NIXL, 2025]](#12-references).

This is a multi-node optimization opportunity because the prefill and decode pools are typically on separate nodes, and the KV cache transfer dominates the inter-node communication. The transfer can be pipelined layer-by-layer rather than waiting for full prefill completion.

#### 3.2.5 Optimization 4: Network-Level Protocol Selection

**The problem:** Different network fabrics (InfiniBand, RoCE, Slingshot, Ethernet) have different optimal protocols for different message sizes and communication patterns.

**The optimization:** NCCL provides several network-level optimizations:

- **Direct NIC support** (NCCL 2.27): Enables GPUs to access network interfaces directly, utilizing full network bandwidth without CPU-mediated staging [[NCCL 2.27 Blog]](#12-references).
- **GIN (GPU-Initiated Networking)** (NCCL 2.28): Allows CUDA kernels to initiate network operations directly, eliminating host-initiated synchronization overhead [[NCCL 2.28 Blog]](#12-references).
- **UCX transport selection:** For disaggregated serving, TensorRT-LLM selects UCX transport layers based on available hardware (`UCX_TLS` configuration).

TensorRT-LLM's role is primarily to select the right AllReduce strategy (via the AutoTuner or lookup table) and let NCCL handle the network-level optimization internally. The exception is MNNVL, which uses `ncclMemAlloc` + fabric handles to bypass NCCL's generic network path for NVSwitch-fabric-connected nodes.

### 3.3 Cross-Cutting Optimizations (Both Intra- and Inter-Node)

#### 3.3.1 Runtime AutoTuning

**The problem:** The optimal AllReduce strategy depends on message size, hardware topology, GPU generation, and even the specific NVLink/NVSwitch firmware. Static heuristics cannot capture all these variables.

**The optimization:** TensorRT-LLM's AutoTuner benchmarks all valid strategies at runtime for each unique input shape and caches the winner:

```python
# torch_custom_ops.py — AllReduceRunner enumerates and benchmarks tactics
def get_valid_tactics(self, ...):
    valid_strategies = [
        AllReduceStrategy.NCCL_SYMMETRIC.value,
        AllReduceStrategy.NCCL.value,
    ]
    if workspace_size <= max_workspace_size:
        valid_strategies.append(AllReduceStrategy.ONESHOT.value)
    if inputs[0].shape[0] >= self.tp_size:
        valid_strategies.append(AllReduceStrategy.TWOSHOT.value)
    return valid_strategies
```

The AutoTuner discovers opportunities that static tables miss — for example, on B200 TP=4 with 1024 tokens, NCCL_SYMMETRIC's hardware multicast beats TWOSHOT fusion despite requiring separate norm kernels. The static lookup table would select TWOSHOT; the AutoTuner correctly selects NCCL_SYMMETRIC (measured in nsys profiling, see [Section 9](#9-profiling-results-measured-behavior-on-b200)).

**Trade-off:** The first call per shape incurs benchmarking warmup. Disable with `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` for deterministic profiling.

#### 3.3.2 Empirically-Tuned Strategy Lookup Tables

When the AutoTuner is disabled, a per-SM-generation lookup table provides the strategy:

```cpp
// customAllReduceUtils.h — two tables, indexed by [TP][fusion_op][hidden_size][tokens]
AllReduceBestStrategyTableSM90   // H100 — profiled on H100 DGX
AllReduceBestStrategyTableSM100  // B200 — profiled on B200 DGX
```

Key insight: the same workload requires different strategies on different hardware. H100 TP=8 falls to NCCL at just 64 tokens (SM90 NVLink can't handle the write amplification), while B200 TP=8 keeps ONESHOT up to 128 tokens (SM100's faster NVLink shifts the crossover).

#### 3.3.3 Workspace Size as a Strategy Boundary

The 64 MiB default workspace (`TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE`) creates a hard boundary between custom kernels and NCCL:

```
message_size = total_tokens × hidden_size × dtype_size

BF16, hidden=8192:
  4096 tokens: 4096 × 8192 × 2 = 64 MiB  → at the limit (custom kernel eligible)
  8192 tokens: 8192 × 8192 × 2 = 128 MiB → exceeds limit → NCCL fallback
```

For high-batch decode scenarios (e.g., 8192 concurrent requests), the workspace limit forces NCCL fallback. Increasing the workspace size via the environment variable recovers custom kernel eligibility at the cost of GPU memory.

### 3.4 Summary: Optimization Landscape

| Optimization | Scope | Mechanism | Benefit | TRT-LLM Status |
|-------------|-------|-----------|---------|----------------|
| **Direct IPC kernels** | Intra-node | Bypass NCCL, direct pointer dereference | 2–5× lower latency for small msgs | Implemented (ONESHOT/TWOSHOT) |
| **AllReduce + Norm fusion** | Intra-node (custom) | Single kernel, data in registers | 50% less HBM traffic, 2–3 fewer launches | Implemented |
| **NVSwitch multicast** | Intra-node | LDMC/STMC hardware instructions | O(1) communication, fastest raw AR | Implemented (NCCL_SYMMETRIC, SYMM_MEM) |
| **GEMM+AllReduce fusion** | Intra-node | GEMM writes to comm buffer directly | Eliminates GEMM→HBM→AR copy | Implemented (SM100+, opt-in) |
| **Completion overlap** | Intra-node | Deferred signal write, stream serialization | Overlaps signal with next compute | Implemented |
| **Low-precision comm** | Intra-node (PCIe) | FP8 quantized P2P exchange | 2× less PCIe traffic | Implemented (LOWPRECISION) |
| **Heterogeneous link aggregation** | Intra-node | NVLink + PCIe + NIC combined | 27% more bandwidth | Not implemented (research) |
| **MNNVL** | Inter-node (NVL72) | Fabric-managed multicast buffers | True AR across nodes without IPC | Implemented (RESIDUAL_RMS_NORM only) |
| **Hierarchical AllReduce** | Inter-node | Intra-node reduce → inter-node AR | Reduces cross-node traffic | Via NCCL internal (PAT algorithm) |
| **Disaggregated serving** | Inter-node | Separate prefill/decode, KV transfer | Better GPU utilization, pipeline overlap | Implemented (NIXL, UCX) |
| **Network-level protocols** | Inter-node | Direct NIC, GPU-initiated networking | Full network bandwidth utilization | Via NCCL 2.27+/2.28+ |
| **Runtime AutoTuning** | Both | Benchmark all strategies per shape | Discovers hardware-specific optima | Implemented (default) |
| **Per-SM lookup tables** | Both | Static empirically-profiled tables | Fast fallback, no warmup cost | Implemented (SM90, SM100) |

---

## 4. Meet the Model: Nemotron-H

[Nemotron-H](https://research.nvidia.com/labs/adlr/nemotronh/) is a family of hybrid Mamba-Transformer models from NVIDIA, available in 8B, 47B, and 56B parameter variants. The key innovation is replacing the majority of self-attention layers with Mamba layers that use constant memory during generation (versus the linearly-growing KV cache of attention), achieving up to 3× faster inference than comparably-sized pure transformer models [[Nemotron-H Paper, 2025]](#12-references).

**Nemotron-H Ultra** (the largest variant we analyze) has 108 layers with three distinct types:

| Layer Type | Count | Subcomponents | AllReduce Points |
|-----------|------:|---------------|-----------------|
| **Mamba** (M) | 48 | SSM mixer + MLP | 1 per layer (after MLP `down_proj`) |
| **MoE** (E) | 48 | Attention + Mixture-of-Experts | 1 per layer (after MoE combine) |
| **Transformer** (\*) | 12 | Attention + dense MLP | 2 per layer (after `o_proj` + after `down_proj`) |

This gives **120+ AllReduce operations per forward pass** — making AllReduce optimization critical for inference latency.

What makes Nemotron-H particularly interesting for AllReduce fusion is that **all three layer types** participate in the same fusion pattern. Despite having different computational kernels (SSM vs attention, dense MLP vs MoE routing), the post-mixer → residual → norm pipeline is identical. This uniformity enables a single cross-layer fusion strategy to cover the entire model.

---

## 5. AllReduce Strategies in TensorRT-LLM

TensorRT-LLM implements a rich set of AllReduce strategies, each optimized for different message sizes, hardware topologies, and fusion requirements. The strategy enum lives in both Python (`tensorrt_llm/functional.py`) and C++ (`customAllReduceKernels.h`):

```cpp
enum class AllReduceStrategyType : int8_t {
    NCCL = 0, MIN_LATENCY = 1, UB = 2, AUTO = 3,
    ONESHOT = 4, TWOSHOT = 5, LOWPRECISION = 6, MNNVL = 7, NCCL_SYMMETRIC = 8,
};
// SYMM_MEM = 9 exists as a Python-level strategy
```

### 5.1 Strategy Overview

| Strategy | Communication Mechanism | True Fusion | Requires | Best For |
|----------|------------------------|:-----------:|----------|----------|
| **ONESHOT** | IPC push (Lamport protocol) | Yes | NVLink + P2P | Decode (≤128 tokens) |
| **TWOSHOT** | IPC reduce-scatter + allgather | Yes | NVLink + P2P | Decode/Prefill (129–4K tokens) |
| **MIN_LATENCY** | Auto-selects ONESHOT or TWOSHOT | Yes | NVLink + P2P | General custom kernel |
| **NCCL** | NCCL library (ring/tree) | No | Any topology | Large messages, universal fallback |
| **NCCL_SYMMETRIC** | NCCL + window registration | No | NVLink preferred | Large messages on NVLink |
| **SYMM_MEM** | PyTorch MULTIMEM instructions | No | SM ≥ 9.0, 4–8 GPUs | Raw allreduce (no fusion) |
| **MNNVL** | Multicast NVLink hardware | Partial | GB200 NVL72, multi-node | Multi-node NVLink |
| **UB** | User Buffers (zero-copy) | Own kernels | torch.compile, SM ≥ 9.0 | GEMM→AllReduce overlap |
| **AUTO** | Selects best at runtime | Depends | — | Default, recommended |

The remainder of this section examines each strategy in detail.

### 5.2 Custom IPC Kernels: ONESHOT and TWOSHOT

These are TensorRT-LLM's custom CUDA kernels that bypass NCCL entirely, using **IPC-mapped GPU memory** for direct GPU-to-GPU communication over NVLink. They are the only strategies that support **true single-kernel fusion** — AllReduce + residual + RMSNorm + optional quantization, all in one `cudaLaunchKernelEx` call.

#### 4.2.1 ONESHOT: The Lamport Protocol

The ONESHOT kernel uses a barrier-free synchronization technique inspired by Lamport's work on concurrent programming [[Lamport, 1977]](#12-references). The core idea: instead of explicit barriers, use **sentinel values** to detect when data has arrived.

**Algorithm:**

```
ONE KERNEL LAUNCH
│
├── PHASE 1: Push (no barrier — just write)
│   Each rank writes its full tensor to ALL ranks' IPC buffers.
│   Rank 0 → buffer[0], buffer[1], ..., buffer[N-1]
│   Rank 1 → buffer[0], buffer[1], ..., buffer[N-1]
│   ... (all ranks in parallel)
│
│   After push, Rank 0's LOCAL buffer contains:
│   slot[0]: [data from Rank 0]  ← local write
│   slot[1]: [data from Rank 1]  ← arrived over NVLink
│   slot[2]: [data from Rank 2]  ← arrived over NVLink
│   slot[3]: [data from Rank 3]  ← arrived over NVLink
│
├── PHASE 2: Spin-wait + Sum + Fuse (no barrier)
│   Each thread polls its LOCAL buffer:
│     while (slot[r] == NEGATIVE_ZERO_SENTINEL) { }   // spin until data arrives
│   Sums all N slots (local reads only — no NVLink traffic)
│   Applies fused post-processing: residual add → RMSNorm → optional quant
│
└── Done (update triple-buffer flag)
```

**Synchronization mechanism:** The workspace buffers are pre-initialized with **negative zero** (`0x80000000` in IEEE 754 float32) as a sentinel value. When a rank writes real data to another rank's buffer, the sentinel is overwritten. The receiving rank's threads spin-wait using volatile loads, checking each slot until the sentinel disappears — meaning real data has arrived. No barriers, no atomics, no NCCL.

**Triple-buffering:** To avoid the ABA problem (where a new iteration's data could be mistaken for the previous iteration's), 3 buffer slots are rotated. Each kernel invocation uses `flag_value % 3` and clears `(flag_value + 2) % 3`. The workspace cost is `3 × N × max_tokens × hidden_size × sizeof(dtype)`.

**NVLink traffic:** Each rank writes its data to all `N-1` remote ranks' buffers. Total traffic per rank: `(N-1) × D` writes, where `D = total_tokens × hidden_size × dtype_size`. This is **O(N) write amplification** — the fundamental trade-off of ONESHOT.

**Why it wins for small messages:** For 1 decode token with `hidden_size=8192` in bf16, `D = 16 KB`. At 900 GB/s NVLink bandwidth, transferring 16 KB takes ~18 ns. Even with 8× write amplification (TP=8), the total is 128 KB = ~0.14 µs — negligible compared to kernel launch overhead. ONESHOT's barrier-free design means each thread begins processing the moment its data arrives, with no synchronization stalls.

#### 4.2.2 TWOSHOT: Explicit-Barrier Reduce-Scatter + AllGather

The TWOSHOT kernel uses the classic reduce-scatter → allgather decomposition, but within a **single CUDA kernel**, using flag-based barriers for synchronization.

**Algorithm:**

```
ONE KERNEL LAUNCH
│
├── PHASE 1: Write own data to own buffer (local write, no NVLink)
│
├── ★ BARRIER 1 ★ (flag-based acquire/release, all ranks must reach)
│
├── PHASE 2: Reduce-scatter
│   Tokens partitioned: Rank 0 → tokens[0..T/N-1], Rank 1 → tokens[T/N..2T/N-1], ...
│   Each rank reads its assigned chunk from ALL N ranks' buffers (NVLink reads)
│   Sums them, writes partial results to ALL N ranks' buffers (NVLink writes)
│
├── ★ BARRIER 2 ★
│
├── PHASE 3: AllGather + Fuse
│   Each rank reads ALL reduced chunks from LOCAL buffer (no NVLink)
│   Applies fused post-processing: residual add → RMSNorm → optional quant
│
└── Done (update barrier flag)
```

**Synchronization mechanism:** Two explicit barriers using PTX `st.global.release.sys` / `ld.global.acquire.sys` instructions. These provide memory ordering guarantees across GPU ranks.

**NVLink traffic comparison with ONESHOT:**

| Metric | ONESHOT | TWOSHOT |
|--------|---------|---------|
| Remote NVLink writes per rank | `(N-1) × D` | `(N-1) × D/N` |
| Remote NVLink reads per rank | 0 (read local) | `(N-1) × D/N` |
| Total NVLink traffic (all ranks) | `N × (N-1) × D` | `2 × (N-1) × D` |
| Write amplification | O(N) | O(1) |

For TP=8 with `D=2 MB`: ONESHOT generates 112 MB total NVLink traffic; TWOSHOT generates 28 MB (4× less).

**Why TWOSHOT wins for larger messages:** As token counts grow beyond ~128, ONESHOT's O(N) write amplification begins saturating NVLink bandwidth. At 512 tokens with TP=8 (`D = 512 × 8192 × 2 = 8 MB`), ONESHOT would write 448 MB across all ranks. TWOSHOT's 2 barrier syncs (~2–10 µs total) become negligible relative to the saved bandwidth. Additionally, ONESHOT's workspace requirement of `N × D` per rank approaches the default 64 MiB workspace limit much faster.

**The crossover point** is empirically determined at `kOneShotMaxToken = 128`:

```cpp
// allReduceFusionKernels.h
static constexpr int kOneShotMaxToken = 128;
```

The `MIN_LATENCY` meta-strategy auto-selects between them:

```cpp
// allreduceOp.cpp
allreduce_fusion_params.use_oneshot = seq_len <= kOneShotMaxToken
    || hidden_size < static_cast<int64_t>(tp_size);
```

### 5.3 NCCL-Based Strategies: NCCL, NCCL_SYMMETRIC, SYMM_MEM

These three strategies all leverage NVIDIA's multi-GPU hardware, but through different software stacks and at different abstraction levels:

```
                     ┌──────────────────────────────────┐
  SYMM_MEM           │  PyTorch symmetric memory API     │ ← PyTorch manages buffers
                     │  multimem_all_reduce_() directly  │ ← calls MULTIMEM HW instruction
                     └──────────────────────────────────┘
                     ┌──────────────────────────────────┐
  NCCL_SYMMETRIC     │  NCCL library                     │ ← NCCL manages allreduce
                     │  ncclMemAlloc + Window Register   │ ← NCCL-managed symmetric buffers
                     │  ncclAllReduce() — may use MULTIMEM internally
                     └──────────────────────────────────┘
                     ┌──────────────────────────────────┐
  NCCL (plain)       │  NCCL library                     │ ← NCCL manages everything
                     │  Regular cudaMalloc buffers       │ ← NCCL stages through internal bufs
                     │  ncclAllReduce() — ring/tree algo │
                     └──────────────────────────────────┘
```

#### NCCL (plain)

The universal fallback. Calls `ncclAllReduce()` with regular GPU tensors. NCCL copies data into internal buffers and runs ring or tree algorithms. Works on any topology (NVLink, PCIe, InfiniBand, Ethernet). No special setup overhead, but internal staging adds hidden latency.

When used with fused operations (e.g., `RESIDUAL_RMS_NORM`), the allreduce and norm run as **separate kernels** — this is "pseudo-fusion" where the runtime dispatches `ncclAllReduce()` followed by a standalone `residualRmsNorm` kernel. The intermediate reduced tensor hits HBM between the two kernels.

#### NCCL_SYMMETRIC

Introduced in [NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27), this adds "window registration": buffers are allocated with `ncclMemAlloc` and registered via `ncclCommWindowRegister()`. This tells NCCL the buffer is directly accessible by all ranks via NVSwitch, allowing it to skip internal staging and potentially use hardware multicast (LDMC/STMC) internally.

The trade-off is one explicit `memcpy` to copy user input into the registered window buffer. An empirically-tuned heuristic decides when this is worthwhile:

```cpp
// allreduceOp.cpp — linear model fitted on GB200
double const a = -4986.43478503;
double const b = 156716.52177552;
size_t minRegistrationThreshold = max(0, a * nRanks + b) * element_size;
// TP=2: ~293 KB, TP=4: ~273 KB, TP=8: ~234 KB
```

Below the threshold, registration overhead exceeds the benefit, and it behaves like plain NCCL.

NCCL_SYMMETRIC can achieve **up to 9× latency reduction for small messages** and **2.5× higher throughput for medium messages** on NVL8 systems [[NCCL 2.27 Blog]](#12-references).

#### SYMM_MEM (PyTorch Symmetric Memory + MULTIMEM)

Bypasses NCCL entirely. Uses PyTorch's `torch.distributed._symmetric_memory` API to allocate multicast-capable memory, then calls `torch.ops.symm_mem.multimem_all_reduce_()` — which directly issues MULTIMEM hardware instructions on the NVSwitch fabric.

MULTIMEM is the most direct path to the hardware. It atomically reads, sums, and writes across all GPUs in a **single hardware operation** — O(1) complexity, no barriers, no software algorithm:

| Strategy | Mechanism | Complexity |
|----------|-----------|-----------|
| NCCL | Ring/tree algorithm | O(N) steps |
| ONESHOT | Push to all, spin-wait | O(N) write amplification |
| TWOSHOT | Reduce-scatter + allgather | O(1) amplification, 2 barriers |
| **SYMM_MEM** | `multimem_all_reduce_()` | **O(1), no barriers, hardware** |

**The critical limitation:** SYMM_MEM only supports `fusion_op=NONE`. MULTIMEM is a hardware reduction instruction — it can sum values, but it has no concept of post-processing. It cannot also add a residual, compute RMSNorm, or quantize to FP4. When fused operations are requested, SYMM_MEM is **skipped entirely**:

```python
# AllReduce.forward()
if self.symm_mem_allreduce and all_reduce_params.fusion_op == AllReduceFusionOp.NONE:
    return self.symm_mem_allreduce(input)       # ← used for plain allreduce
elif self.symm_mem_allreduce and all_reduce_params.fusion_op != AllReduceFusionOp.NONE:
    pass  # ← SKIPPED, falls through to NCCL/custom kernel path
```

In Nemotron-H with `fuse_allreduce_norm=True`, every allreduce has `fusion_op=RESIDUAL_RMS_NORM` or a quant variant — so SYMM_MEM is never used for the model layers. This means enabling SYMM_MEM does not help when AllReduce+norm fusion is active.

### 5.4 Multi-Node NVLink: MNNVL

MNNVL (Multi-Node NVLink) is a strategy for GB200 NVL72-class systems where the TP group spans multiple OS nodes connected via NVSwitch fabric. On these systems, CUDA IPC (used by ONESHOT/TWOSHOT) is unavailable across node boundaries — there is no shared virtual address space between GPUs on different nodes.

MNNVL solves this by using `ncclMemAlloc` with NVSwitch fabric handles instead of CUDA IPC. Its kernel uses a Lamport-style one-shot/two-shot split similar to the custom IPC kernels:

- Supports true fusion for `RESIDUAL_RMS_NORM`
- Does **not** support NVFP4 or FP8 quantization fusion (implementation gap — see [Section 10](#10-gaps-trade-offs-and-future-directions))
- Only activates on multi-node aarch64 (Grace CPU) platforms with NVSwitch fabric

In practice, MNNVL is the GB200 NVL72 with TP=8 spanning 2 nodes — the only production configuration where it activates.

### 5.5 User Buffers (UB): Zero-Copy GEMM-to-AllReduce

UB is a fundamentally different approach. Instead of optimizing the AllReduce kernel itself, UB eliminates the copy between GEMM output and the communication buffer.

With standard ONESHOT/TWOSHOT, the data flow is:

```
GEMM output → HBM → read into kernel → push to IPC buffers
```

With UB:

```
GEMM output → directly written to pre-registered IPC buffer → AllReduce from there
```

This is achieved through `torch.compile` graph pattern matching: the compiler finds `GEMM → AllReduce` sequences in the computation graph and rewrites them so the GEMM writes directly to a User Buffer. UB has its own fused kernels supporting residual + RMSNorm + NVFP4 quantization.

**UB requires torch.compile**, `pp_size==1`, and is **mutually exclusive** with the cross-layer fusion approach analyzed in this post. The cross-layer fusion disables per-layer AllReduce (`enable_allreduce=False`), breaking the `GEMM → AllReduce` pattern that torch.compile needs to match. The two approaches target different deployment scenarios:

| Deployment | Recommended Approach |
|-----------|---------------------|
| Eager mode (no torch.compile) | Cross-layer fused path |
| torch.compile + PP=1 | Unfused path + UB |
| PP > 1 | Cross-layer fused path (only option) |

### 5.6 Strategy Comparison Summary

For the most common fusion pattern (`RESIDUAL_RMS_NORM`), here is what each strategy actually does:

| Strategy | What Happens | Kernel Launches | Data Path |
|----------|-------------|:--------------:|-----------|
| **ONESHOT** | Single fused kernel | **1** | IPC push → sum in registers → residual → norm → write |
| **TWOSHOT** | Single fused kernel | **1** | Reduce-scatter → barrier → allgather → residual → norm → write |
| **MNNVL** | Single fused kernel | **1** | Multicast push → sum in registers → residual → norm → write |
| **UB** | Own fused kernel | **1** | GEMM → UB → allreduce + residual + norm → write |
| **NCCL** | Allreduce then norm | **2–3** | `ncclAllReduce` → HBM → `residualRmsNorm` → HBM |
| **NCCL_SYMMETRIC** | Faster allreduce then norm | **2–3** | Symmetric allreduce → HBM → `residualRmsNorm` → HBM |
| **SYMM_MEM** | Skipped for fusion | **N/A** | Only handles `fusion_op=NONE` |

---

## 6. The Core Optimization: Fusing AllReduce with RMSNorm

### 6.1 Why Fusion Matters: The Memory Bandwidth Argument

AllReduce, residual addition, and RMSNorm are all **memory-bandwidth-bound** operations on GPU HBM. They perform minimal arithmetic per byte of data moved. When run as separate CUDA kernels, each one reads the full tensor from HBM, does a small amount of work, and writes it back.

**Without fusion — 3 to 4 separate kernels:**

```
HBM → [AllReduce kernel] → HBM → [Residual Add kernel] → HBM → [RMSNorm kernel] → HBM
       ↑ read    ↓ write    ↑ read        ↓ write         ↑ read       ↓ write
                          6 HBM transactions total
```

**With fusion — 1 kernel:**

```
HBM → [AllReduce + Residual + RMSNorm + optional Quant] → HBM
       ↑ read                                     ↓ write
                     2 HBM transactions total
```

The fused kernel keeps intermediate data in **GPU registers** throughout the entire pipeline. Data flows from the IPC communication buffer directly into registers, where residual addition, RMSNorm, and optional quantization are applied before a single write to HBM.

Three sources of savings:

1. **Eliminated HBM round-trips.** For `hidden_size=8192` in bf16, each eliminated round-trip saves `8192 × 2 = 16 KB` per token × 2 (read+write) = 32 KB per token per round-trip. The fused kernel eliminates 2–3 such round-trips.

2. **Reduced kernel launch overhead.** Each CUDA kernel launch has ~5–10 µs overhead. Across 108 layers, eliminating 2–3 launches per layer saves 216–324 µs per forward pass.

3. **Intra-kernel communication/compute overlap.** In the ONESHOT Lamport kernel, residual addition and RMSNorm happen on data *as it arrives from peer GPUs*. Communication and post-processing overlap within the same kernel.

### 6.2 How It Works in Nemotron-H

The fusion architecture uses a **cross-layer** pattern. Instead of each layer doing its own AllReduce followed by norm, each layer's `pre_allreduce` handles the AllReduce of the **previous** layer's mixer output, fused with the current layer's norm.

**Before fusion (per-layer pattern):**

```
Layer N:
  mixer forward → AllReduce → residual add → RMSNorm → output

Layer N+1:
  mixer forward → AllReduce → residual add → RMSNorm → output
```

**After fusion (cross-layer pattern):**

```
Layer N:
  mixer forward → (unreduced partial sum, skip AllReduce)

Layer N+1:
  pre_allreduce [AllReduce(Layer N output) + residual + RMSNorm] → mixer forward → ...
```

Layer 0 skips `pre_allreduce` (the embedding output is already fully reduced). A `final_allreduce` after the last layer handles the final layer's output → final norm.

All mixer types participate uniformly:
- **Mamba / MLP layers**: Disable their own AllReduce at init time (`reduce_output=False`)
- **Transformer / MoE layers**: Disable their own AllReduce at forward time (`AllReduceParams(enable_allreduce=False)`)

This is implemented in the Nemotron-H model class (`modeling_nemotron_h.py`):

```python
# At init time:
if self.fuse_allreduce_norm and layer_idx > 0:
    self.pre_allreduce = AllReduce(mapping, strategy)

# At forward time:
if hasattr(self, 'pre_allreduce'):
    if norm.is_nvfp4:
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
    else:
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

    all_reduce_params = AllReduceParams(
        fusion_op=fusion_op,
        residual=residual,
        norm_weight=norm.weight,
        eps=norm.variance_epsilon,
    )
    result = self.pre_allreduce(hidden_states, all_reduce_params=all_reduce_params)

# Tell mixer to skip its own AllReduce:
mixer_kwargs['all_reduce_params'] = AllReduceParams(enable_allreduce=False)
```

### 6.3 Inside the Fused Kernel

The fused kernel is not two separate kernels "glued together." The C++ compiler generates **one monolithic CUDA kernel** with both AllReduce communication and post-processing, wired together via template metaprogramming.

#### Launch path

```
Python: AllReduce.forward(fusion_op=RESIDUAL_RMS_NORM)
  → C++: allreduceOp.run()
    → selectImplementation() → ONESHOT
    → runFusionAllReduce():
        1. Allocate output tensors (norm_out, residual_out)
        2. Fill AllReduceFusionParams with ALL pointers
        3. Set params.pattern = kARResidualRMSNorm
        4. Call allreduce_fusion_op(params)
          → Macro dispatch: DISPATCH_RANKS(4) → DISPATCH_DTYPE(bf16) → DISPATCH_PATTERN
          → allreduce_fusion_kernel_launcher<kARResidualRMSNorm, bf16, 4, false>(params)
            → cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_oneshot_lamport<...>, params)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       ONE cudaLaunchKernelEx — ONE __global__ function
```

#### Template-based fusion: `FusedOp<Pattern, DType>`

The template parameter `Pattern` (e.g., `kARResidualRMSNorm`) controls which operations are compiled into the kernel via `if constexpr` — evaluated at compile time, zero runtime overhead:

```cpp
__device__ __forceinline__ void operator()(float4 val, int token_id) {
    // Step 1: Residual add (compiled-in if Pattern requires it)
    if constexpr (HasResidual<Pattern>) {
        val = add128(val, residual);
        if constexpr (HasResidualOut<Pattern>)
            store(residual_out, val);
    }

    // Step 2: RMSNorm (compiled-in if Pattern requires it)
    if constexpr (HasRMSNorm<Pattern>) {
        val = rms_norm(val, gamma);
        if constexpr (HasNormOut<Pattern>)
            store(norm_out, val);
    }

    // Step 3: Quantize (compiled-in if Pattern requires it)
    if constexpr (GetQuantType<Pattern> == QuantType::kFP4)
        quantize_and_store_fp4(val);
    else if constexpr (GetQuantType<Pattern> == QuantType::kFP8)
        quantize_and_store_fp8(val);
}
```

For `kARResidualRMSNorm`, the compile-time traits are `{HasResidual=true, HasResidualOut=true, HasRMSNorm=true, HasNormOut=true, QuantType=kNone}`. The compiler emits: residual add → store residual → RMSNorm → store norm output. The quantize branches are compiled away entirely.

The kernel's main loop calls this `FusedOp` immediately after the AllReduce sum:

```cpp
// Inside the ONESHOT kernel
float4 sum_val = allreduce_sum<DType, NRanks>(vals);   // AllReduce sum
fused_op(sum_val, tidx);                                // residual + norm + quant
```

Seven fusion patterns are compiled as separate kernel instantiations:

| Pattern | Operations | Use Case |
|---------|-----------|----------|
| `kAllReduce` | AllReduce only | Plain reduce, no fusion |
| `kARResidualRMSNorm` | AR + Residual + Norm | Standard fused path |
| `kARResidualRMSNormFP8Quant` | AR + Residual + Norm + FP8 | FP8 quantized models |
| `kARResidualRMSNormFP4Quant` | AR + Residual + Norm + NVFP4 | NVFP4 quantized models |
| `kARResidualRMSNormOutFP8Quant` | AR + Residual + Norm (out) + FP8 | Norm output + FP8 |
| `kARResidualRMSNormOutFP4Quant` | AR + Residual + Norm (out) + NVFP4 | MoE layers (need norm out for routing) |
| `kARRMSNorm` | AR + Norm (no residual) | Special cases |

### 6.4 Memory Traffic Analysis

Concrete numbers for `hidden_size=8192`, bf16, per-token data size = `8192 × 2 = 16 KB`:

**Unfused path (NCCL + separate kernels):**

| Kernel | HBM Reads | HBM Writes |
|--------|-----------|------------|
| `ncclAllReduce` | 16 KB (partial_sum) | 16 KB (reduced) |
| `residualRmsNorm` | 16 KB (reduced) + 16 KB (residual) | 16 KB (norm_out) + 16 KB (new_residual) |
| **Total per token** | **48 KB** | **48 KB** |
| **Total HBM traffic** | | **96 KB** |

**Fused path (ONESHOT, data in registers):**

| Kernel | HBM Reads | HBM Writes |
|--------|-----------|------------|
| `oneshot_lamport<kARResidualRMSNorm>` | 16 KB (residual) | 16 KB (new_residual) + 16 KB (norm_out) |
| **Total per token** | **16 KB** | **32 KB** |
| **Total HBM traffic** | | **48 KB** |

*(The partial_sum and reduced tensor never hit HBM — they flow through IPC buffers and registers.)*

**Savings: 96 KB → 48 KB per token = 50% reduction in HBM traffic.**

For a decode batch of 64 tokens across 108 layers:
- Saved per forward pass: `(96 - 48) × 64 × 108 = 331,776 KB ≈ 324 MB`
- Plus: 108 × 2 = 216 fewer kernel launches (at ~5–10 µs each = ~1–2 ms saved)

### 6.5 NVFP4 Quantization Fusion

When NVFP4 quantization is active (common on Blackwell GPUs for inference), the fusion extends further. Instead of the 3-kernel unfused path:

```
ncclAllReduce → residualRmsNorm → fp4_quantize   (3 kernels, 3 HBM round-trips)
```

The fused kernel computes:

```
allreduce → residual add → RMSNorm → NVFP4 quant   (1 kernel, data in registers)
```

The quantization step (`kARResidualRMSNormFP4Quant`) converts the bf16 norm output to NVFP4 format directly in registers before the final write to HBM. This saves an additional kernel launch and HBM round-trip compared to even the `RESIDUAL_RMS_NORM` pattern.

In Nemotron-H with NVFP4, different layers use different patterns:
- **Mamba/MLP layers** (pattern 3): `kARResidualRMSNormFP4Quant` — norm output goes directly to the next layer's NVFP4 input
- **MoE layers** (pattern 5): `kARResidualRMSNormOutFP4Quant` — norm output is also needed in bf16 for the MoE router, so both bf16 and NVFP4 outputs are produced
- **Transformer layers** (pattern 1): `kARResidualRMSNorm` — feeding non-NVFP4 paths

---

## 7. Strategy Selection: From AUTO to the Right Kernel

### 7.1 The Two-Layer Decision Process

When `allreduce_strategy=AUTO` (the default), strategy selection happens in **two layers** — Python dispatches first, then C++ makes the final decision.

**Layer 1: Python (`AllReduce.forward()` in `ops.py`)**

```python
# Priority 1: Try SYMM_MEM (only if fusion_op is NONE)
if self.symm_mem_allreduce and fusion_op == NONE:
    return self.symm_mem_allreduce(input)            # MULTIMEM hardware

# Priority 2: Try MNNVL (only if multi-node NVLink)
if self.mnnvl_allreduce:
    result = self.mnnvl_allreduce(input, all_reduce_params)
    if result is not None:
        return result                                 # handles NONE and RESIDUAL_RMS_NORM

# Priority 3: Fall through to C++ allreduceOp
output = torch.ops.trtllm.tunable_allreduce(input, ..., strategy=AUTO, ...)
```

**Layer 2: C++ (`allreduceOp.cpp`)**

The C++ layer uses either the **AutoTuner** (default) or a **static lookup table** to select the specific strategy.

### 7.2 The Static Lookup Table

Two empirically-profiled lookup tables exist: `AllReduceBestStrategyTableSM90` (H100) and `AllReduceBestStrategyTableSM100` (B200/GB200). They are indexed by:

```cpp
tp_index         = log2(tp_size) - 1         // TP=2→0, TP=4→1, TP=8→2
fusion_op_index  = mapFusionOpToIndex[op]     // NONE→0, RMS_NORM→1, FP8→2, NVFP4→3
hidden_size_index = log2(hidden_size) - 7     // 128→0, 256→1, ..., 8192→6
num_token_index  = log2(num_tokens)           // 1→0, 2→1, 4→2, ..., 8192→13
```

Each entry maps to `4=ONESHOT`, `5=TWOSHOT`, or `0=NCCL`.

**Example: SM100 (B200), RESIDUAL_RMS_NORM, bf16, hidden=8192:**

| Tokens | TP=2 | TP=4 | TP=8 |
|-------:|:----:|:----:|:----:|
| 1 | ONESHOT | ONESHOT | ONESHOT |
| 16 | ONESHOT | ONESHOT | ONESHOT |
| 64 | ONESHOT | ONESHOT | ONESHOT |
| 128 | ONESHOT | ONESHOT | ONESHOT |
| 256 | ONESHOT | **TWOSHOT** | **TWOSHOT** |
| 512 | ONESHOT | **TWOSHOT** | **TWOSHOT** |
| 1K | ONESHOT | **TWOSHOT** | **TWOSHOT** |
| 4K | **NCCL** | **NCCL** | **TWOSHOT** |
| 8K | **NCCL** | **NCCL** | **NCCL** |

Key observation: B200 (SM100) keeps custom fused kernels active much more aggressively than H100 (SM90). On H100 with TP=8, the lookup table falls to NCCL at just 64 tokens — the 8× write amplification of ONESHOT makes it uncompetitive on SM90's slower NVLink. B200's faster NVLink (1.8 TB/s vs 0.9 TB/s) shifts the crossover dramatically.

### 7.3 The AutoTuner

When the AutoTuner is enabled (default), it **bypasses the static lookup table entirely**. Instead, `tunable_allreduce` creates an `AllReduceRunner` that benchmarks all valid strategies at runtime for each input shape:

1. Enumerate candidates: always `NCCL` and `NCCL_SYMMETRIC`; add `ONESHOT` if message fits workspace; add `TWOSHOT` if `tokens ≥ tp_size`
2. `AutoTuner.choose_one()` benchmarks each candidate and caches the winner per shape
3. The winning strategy is passed **directly** to the C++ kernel — `selectImplementation()` is bypassed

The AutoTuner has access to `NCCL_SYMMETRIC` as a candidate, which the static lookup table does not. This matters because NCCL_SYMMETRIC can outperform custom kernels for medium-to-large messages by leveraging NVSwitch hardware multicast.

**Trade-off:** The AutoTuner's first call per shape incurs benchmarking warmup. For debugging or profiling, set `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` to use the static table.

### 7.4 Traced Decision Examples

To make the selection process concrete, here are end-to-end traces for real hardware/workload combinations:

#### Example 1: B200 DGX, TP=4, decode batch=64, RESIDUAL_RMS_NORM

```
Python: SYMM_MEM → fusion_op=RESIDUAL_RMS_NORM ≠ NONE → skipped
Python: MNNVL → is_mnnvl()=false (single-node DGX) → skipped
Python: → C++ allreduceOp with strategy=AUTO

C++ selectImplementation(seq_len=64, hidden=8192):
  NCCL fallback? msg=64×8192×2=1 MB < 64 MiB, P2P=true, NVLink=true → No
  Lookup: SM100, TP=4, RESIDUAL_RMS_NORM, hidden=8192, tokens=64
    → ONESHOT (4)

★ Result: ONESHOT — true single-kernel fusion ★
```

#### Example 2: H100 DGX, TP=8, decode batch=64, RESIDUAL_RMS_NORM

```
C++ selectImplementation(seq_len=64, hidden=8192):
  Lookup: SM90, TP=8, RESIDUAL_RMS_NORM, hidden=8192, tokens=64
    → NCCL (0)

★ Result: NCCL — no fusion (fallback) ★
  Why: On H100 TP=8, 8× write amplification makes NCCL faster even at 64 tokens.
```

#### Example 3: B200 DGX, TP=8, same workload as Example 2

```
C++ selectImplementation(seq_len=64, hidden=8192):
  Lookup: SM100, TP=8, RESIDUAL_RMS_NORM, hidden=8192, tokens=64
    → ONESHOT (4)

★ Result: ONESHOT — true fusion ★
  B200's faster NVLink shifts the crossover. Same workload that fell to NCCL on H100 
  gets true fusion on B200.
```

#### Example 4: GB200 NVL72, TP=8 (inter-node), RESIDUAL_RMS_NORM_QUANT_NVFP4

```
Python: MNNVL → mnnvl_allreduce.forward(fusion_op=NVFP4)
  → MNNVL can't handle NVFP4 → return None
Python: → C++ with strategy=AUTO

C++ selectImplementation:
  NCCL fallback? Inter-node → P2P=false → yes

★ Result: NCCL — no fusion ★
  MNNVL lacks NVFP4 fusion, and inter-node means no P2P for custom kernels.
  This is the NVL72 + NVFP4 gap.
```

These examples illustrate a critical insight: **the same workload on different hardware can take completely different AllReduce paths**, and the selection system accounts for this through per-SM lookup tables and runtime autotuning.

---

## 8. Platform Topology: How Hardware Shapes Strategy

### 8.1 P2P vs NVLink vs MNNVL

These are three distinct capabilities, not synonyms:

| Capability | What | Scope | Used By |
|-----------|------|-------|---------|
| **P2P** | CUDA API: one GPU directly dereferences another GPU's memory pointer | Same OS node only | Custom ONESHOT/TWOSHOT kernels (IPC buffers) |
| **NVLink** | Physical high-bandwidth inter-GPU link | Same node (direct or via NVSwitch) | Underlies P2P transport; NCCL optimization |
| **MNNVL** | NVSwitch fabric connecting GPUs across OS nodes | Across nodes via NVSwitch | MNNVL allreduce kernel (multicast buffers) |

**Why P2P is false for inter-node:** Custom ONESHOT/TWOSHOT kernels directly dereference pointers to other GPUs' memory inside CUDA `__global__` functions (e.g., `reinterpret_cast<float4*>(comm.data_bufs[r])[idx]`). This requires `cudaIpcOpenMemHandle` to map remote GPU memory into the local address space. CUDA only supports this for GPUs within the same OS node. Across node boundaries, there is no shared virtual address space — even if NVLink physically connects the GPUs.

```cpp
// allreduceOp.cpp — hardcoded
if (is_inter_node) {
    mIsMNNVLSupported = mIsNVLINKSupported && allRanksMnnvlConnected;
    mIsP2PSupported = false;  // P2P doesn't work across nodes
}
```

**Why MNNVL works despite no P2P:** MNNVL uses a different mechanism entirely. Instead of CUDA IPC shared memory, it allocates multicast-capable memory via `ncclMemAlloc` and communicates through NVSwitch fabric handles. The kernel never directly dereferences another GPU's memory pointer.

### 8.2 Topology Summary: B200 DGX vs NVL72

**B200 DGX (8 GPUs per node):**

| TP | Topology | P2P | NVLink | MNNVL | Best Fused Strategy |
|---:|----------|:---:|:------:|:-----:|---------------------|
| 2 | intra-node | yes | yes | N/A | ONESHOT/TWOSHOT |
| 4 | intra-node | yes | yes | N/A | ONESHOT/TWOSHOT |
| 8 | intra-node | yes | yes | N/A | ONESHOT/TWOSHOT |

All TP sizes fit within a single node → P2P is always available → custom fused kernels are always eligible.

**GB200 NVL72 (4 GPUs per node, 18 nodes):**

| TP | Topology | P2P | NVLink | MNNVL | Best Fused Strategy |
|---:|----------|:---:|:------:|:-----:|---------------------|
| 2 | intra-node | yes | yes | N/A | ONESHOT/TWOSHOT |
| 4 | intra-node | yes | yes | N/A | ONESHOT/TWOSHOT |
| 8 | **inter-node** | **no** | local only | **yes** | MNNVL for RESIDUAL_RMS_NORM; **NCCL for NVFP4** |

TP=8 spans 2 nodes (4 GPUs each) → P2P is unavailable → custom kernels are blocked → MNNVL handles what it can → NCCL catches the rest.

---

## 9. Profiling Results: Measured Behavior on B200

The following profiles were collected on 8× B200 (SM100) running **Nemotron-H Ultra** (108 layers: 48 Mamba + 48 MoE + 12 Transformer, `hidden_size=8192`) with NVFP4 quantization.

### 9.1 Decode Phase: ONESHOT Fusion Confirmed

**Benchmark configuration:** `ISL=1, OSL=1024, concurrency=64, max_batch_size=64` (TP=4)

| Kernel | Fusion Pattern | Count | Total Time | Per-call |
|--------|----------------|------:|----------:|--------:|
| `oneshot_lamport<Pattern=3, bf16, NRanks=4>` | kARResidualRMSNormFP4Quant | 1128 | 14.3 ms | 12.7 µs |
| `oneshot_lamport<Pattern=5, bf16, NRanks=4>` | kARResidualRMSNormOutFP4Quant | 1152 | 9.1 ms | 7.9 µs |
| `oneshot_lamport<Pattern=1, bf16, NRanks=4>` | kARResidualRMSNorm | 312 | 2.5 ms | 8.1 µs |
| `oneshot_lamport<Pattern=0, bf16, NRanks=4>` | kAllReduce (Transformer layers) | 24 | 10.8 ms | 450 µs |

Key observations:
- **Zero `rms_norm_kernel` calls** — norm is fully fused into the ONESHOT kernel
- **Zero `ncclAllReduce` calls** — all AllReduces use the custom fused kernel
- Pattern=3 handles Mamba/MLP layers (NVFP4 quant, no separate norm output)
- Pattern=5 handles MoE layers (NVFP4 quant + bf16 norm output for routing)
- Per-call latency of 7.9–12.7 µs confirms the small-message, latency-sensitive regime where fusion matters most

### 9.2 Prefill Phase: When the AutoTuner Chooses Differently

**Benchmark:** `ISL=1024, OSL=1, concurrency=1, max_batch_size=64` (TP=4)

**With AutoTuner enabled (default):**

| Kernel | Count | Total Time | % of GPU |
|--------|------:|----------:|--------:|
| `ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16` | 2616 | 1161 ms | 59.1% |
| `rms_norm_kernel<bf16, Residual=true>` | 2592 | 39 ms | 2.0% |

All `pre_allreduce` calls fell back to **NCCL_SYMMETRIC** with separate `rms_norm_kernel` — no fusion.

**With AutoTuner disabled (`TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1`):**

| Kernel | Fusion Pattern | Count | Total Time | % of GPU |
|--------|----------------|------:|----------:|--------:|
| `twoshot_sync<Pattern=5, bf16, NRanks=4>` | kARResidualRMSNormOutFP4Quant | 1152 | 643 ms | 34.9% |
| `twoshot_sync<Pattern=3, bf16, NRanks=4>` | kARResidualRMSNormFP4Quant | 1128 | 258 ms | 14.0% |
| `twoshot_sync<Pattern=1, bf16, NRanks=4>` | kARResidualRMSNorm | 312 | 202 ms | 11.0% |

Zero `rms_norm_kernel` calls — full TWOSHOT fusion achieved.

**Why the AutoTuner prefers NCCL_SYMMETRIC for prefill:** At 1024 tokens with TP=4, the message size is `1024 × 8192 × 2 = 16 MiB`. NCCL_SYMMETRIC's `ncclSymkDevKernel` uses NVSwitch hardware multicast (LDMC/STMC) for O(1) communication complexity. Even though it runs norm/quant as separate kernels (3 launches total), the allreduce itself is very fast. The custom TWOSHOT kernel fuses everything into 1 launch, but its P2P communication with explicit barriers is slower than hardware multicast at this message size.

The AutoTuner measures **end-to-end time including all kernel launches** and correctly determines that NCCL_SYMMETRIC's faster communication outweighs TWOSHOT's kernel-launch savings.

This illustrates the trade-off:
- **Decode (≤128 tokens):** ONESHOT fusion wins — communication cost is tiny, kernel launch savings are proportionally large
- **Prefill (1024+ tokens):** NCCL_SYMMETRIC wins — communication dominates, hardware multicast is faster than custom P2P

### 9.3 Decoding the NCCL Kernel Name

The kernel `ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16` encodes its entire algorithm:

```
nccl Symk DevKernel _ AllReduce _ RS x LDMC _ AG x STMC _ sum _ bf16
│    │               │            │   │        │   │       │     │
│    │               │            │   │        │   │       │     └─ bfloat16
│    │               │            │   │        │   │       └─ reduction: sum
│    │               │            │   │        │   └─ Store Multicast
│    │               │            │   │        └─ Phase 2: All-Gather
│    │               │            │   └─ Load Multicast
│    │               │            └─ Phase 1: Reduce-Scatter
│    │               └─ AllReduce collective
│    └─ Symmetric kernel (ncclMemAlloc pre-registered buffers)
└─ NCCL library
```

**Two-phase algorithm using NVSwitch hardware multicast:**

- **Phase 1 — Reduce-Scatter with LDMC:** Each GPU is assigned a `1/N` chunk. It issues a single **multicast load** that reads the corresponding data from **all** peers simultaneously via NVSwitch. One load instruction fetches from N sources. The GPU reduces the loaded values locally.

- **Phase 2 — AllGather with STMC:** Each GPU has a fully-reduced `1/N` chunk. It issues a single **multicast store** that writes to **all** peers simultaneously. NVSwitch replicates the write to N destinations in one operation.

| Metric | Custom TWOSHOT | NCCL Symmetric (LDMC/STMC) |
|--------|---------------|---------------------------|
| Communication | P2P writes + 2 barriers + P2P reads | 1 multicast load + 1 multicast store |
| NVLink traffic per GPU | 2 × D | 2 × D/N (multicast) |
| Synchronization | Explicit software barriers | NVSwitch hardware ordering |
| Kernel launches | 1 (fused) | 3 (allreduce + norm + quant) |

At 16 MiB message size, the 4× NVLink traffic reduction from multicast overcomes the 2 extra kernel launches.

---

## 10. Gaps, Trade-offs, and Future Directions

### Current Gaps

| Gap | Affected Configuration | Root Cause |
|-----|----------------------|------------|
| **NVL72 TP=8 + NVFP4** | All workloads on inter-node TP | MNNVL kernel lacks NVFP4 fusion (implementation gap). P2P=false blocks custom kernels. Falls to NCCL without any fusion. |
| **Prefill with AutoTuner** | B200 TP=4, 1024+ tokens | AutoTuner selects NCCL_SYMMETRIC over TWOSHOT (faster communication outweighs fusion benefit) |
| **Very large batches (8K+ tokens)** | B200/NVL72 TP=8 | Message exceeds 64 MiB workspace or lookup table returns NCCL |
| **SYMM_MEM + fusion** | All configurations with SYMM_MEM enabled | MULTIMEM hardware cannot do post-processing; SYMM_MEM is always skipped for fused ops |

### The MNNVL + NVFP4 Gap

This deserves special attention because GB200 NVL72 is the primary platform for *both* MNNVL and NVFP4. The gap is purely an **implementation limitation**, not a hardware constraint.

Comparing the two kernel codebases:

- **Custom P2P kernel** (`allReduceFusionKernels.cu`): Rich `AllReduceFusionPattern` enum with 7 patterns, templatized `FusionPatternTraits`, full NVFP4 support
- **MNNVL kernel** (`mnnvlAllreduceKernels.cu`): Single `bool rmsNormFusion`, no quantization fields, no quant template parameter

The MNNVL kernel follows the same pattern as the P2P kernel: multicast writes → Lamport sync → accumulate in registers → residual → norm → write. Adding FP4 quantization is additional register-level computation between norm and the final write — exactly the extension the P2P kernel already implements.

### Trade-offs: Fusion vs Hardware Acceleration

The fundamental tension in the strategy landscape is between **kernel fusion** (fewer launches, less HBM traffic) and **hardware-accelerated communication** (MULTIMEM, LDMC/STMC). Currently these are mutually exclusive:

- **Fusion (ONESHOT/TWOSHOT):** Single kernel, all operations in registers, but uses software P2P protocol for communication
- **Hardware acceleration (NCCL_SYMMETRIC, SYMM_MEM):** Fastest communication via NVSwitch multicast, but requires separate kernels for norm/quant

The ideal would combine both: hardware-accelerated allreduce with fused post-processing. This would require either:
- Extending MULTIMEM instructions to support callback-style post-processing (hardware change)
- A hybrid kernel that uses LDMC for the reduce-scatter phase, then fuses norm/quant in the allgather phase (software approach)

### Potential Improvements

1. **Add NVFP4 fusion to MNNVL kernel** — highest impact, unlocks fusion for NVL72 TP=8
2. **Teach AutoTuner to account for fusion savings** — currently it benchmarks end-to-end time, but NCCL_SYMMETRIC's separate norm/quant adds latency that may be underweighted in GEMM-dominated prefill
3. **Increase default workspace size** for high-batch decode scenarios (`TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE` env var exists but default is 64 MiB)
4. **Hybrid multicast + fused norm** kernel — use LDMC/STMC for communication, then fuse norm/quant on the reduced data in a single kernel

---

## 11. Conclusion

AllReduce fusion in TensorRT-LLM is a multi-layered optimization that touches every level of the stack — from CUDA kernel template metaprogramming to Python-level strategy dispatch, from NVLink hardware physics to model architecture design.

The key takeaways:

1. **Fusion is primarily a decode-phase optimization.** When generating tokens one at a time with small batches, AllReduce + norm operations are purely memory-bandwidth-bound. Fusing them into a single kernel that keeps data in registers eliminates ~50% of HBM traffic and 2–3 kernel launches per layer.

2. **The strategy landscape is rich and hardware-dependent.** Nine strategies exist, each optimal for a different combination of message size, GPU generation, topology, and fusion requirement. The same workload that uses ONESHOT fusion on B200 may fall to unfused NCCL on H100.

3. **The selection system is two-layered and adaptive.** Python handles SYMM_MEM and MNNVL priorities; C++ uses either a runtime AutoTuner or static lookup tables profiled per-SM-generation. The AutoTuner can discover that NCCL_SYMMETRIC's hardware multicast beats custom fused kernels for large prefill messages.

4. **Cross-layer fusion enables model-wide coverage.** By shifting AllReduce from the current layer's output to the next layer's input, the fusion pattern applies uniformly across Mamba, Transformer, and MoE layers in Nemotron-H — all three share the same residual + norm pipeline.

5. **Gaps remain at the intersection of features.** The most impactful is NVL72 TP=8 + NVFP4, where neither MNNVL (lacks NVFP4 fusion) nor custom kernels (no P2P across nodes) can provide true fusion. Closing this gap is a matter of kernel implementation, not hardware limitation.

For practitioners deploying LLMs on multi-GPU NVIDIA systems, the actionable advice is:
- Use `allreduce_strategy=AUTO` (the default) — it handles most cases optimally
- For decode-latency-sensitive workloads, ensure `fuse_allreduce_norm=True` is enabled (automatic for TP>1 models)
- For profiling and debugging, use `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` to get deterministic strategy selection
- Monitor nsys traces for `oneshot_lamport` or `twoshot_sync` kernel names to confirm fusion is active

---

## 12. References

1. **NCCL Documentation** — Collective Operations.
   NVIDIA. [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

2. **NCCL GitHub Issues #256, #530** — AllReduce algorithm internals.
   [https://github.com/NVIDIA/nccl/issues/256](https://github.com/NVIDIA/nccl/issues/256), [https://github.com/NVIDIA/nccl/issues/530](https://github.com/NVIDIA/nccl/issues/530)

3. **Patarasuk, P. & Yuan, X. (2009)** — "Bandwidth optimal all-reduce algorithms for clusters of workstations." *Journal of Parallel and Distributed Computing*, 69(2), 117–124.
   [https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)

4. **Li, H. et al. (2024)** — "Optimal, Non-pipelined Reduce-scatter and Allreduce Algorithms." *arXiv:2410.14234*.
   [https://arxiv.org/abs/2410.14234](https://arxiv.org/abs/2410.14234)

5. **Lamport, L. (1977)** — "Concurrent Reading and Writing." *Communications of the ACM*, 20(11), 806–811.
   [https://lamport.azurewebsites.net/pubs/rd-wr.pdf](https://lamport.azurewebsites.net/pubs/rd-wr.pdf)

6. **Shoeybi, M. et al. (2019)** — "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv:1909.08053*.
   [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)

7. **NVIDIA NVLink Blog** — "NVIDIA NVLink and NVIDIA NVSwitch Supercharge Large Language Model Inference."
   [https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/)

8. **NVIDIA GB200 NVL72 Blog** — "NVIDIA GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference."
   [https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)

9. **NCCL 2.27 Blog** — "Enabling Fast Inference and Resilient Training with NCCL 2.27."
   [https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27)

10. **TensorRT-LLM NCCL_SYMMETRIC PR** — Pull Request #4500.
    [https://github.com/NVIDIA/TensorRT-LLM/pull/4500](https://github.com/NVIDIA/TensorRT-LLM/pull/4500)

11. **Nemotron-H Paper (2025)** — "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models."
    [https://arxiv.org/abs/2504.03624](https://arxiv.org/abs/2504.03624)

12. **NVIDIA Nemotron-H Research Page**.
    [https://research.nvidia.com/labs/adlr/nemotronh/](https://research.nvidia.com/labs/adlr/nemotronh/)

13. **Megatron-LM Tensor Parallelism** — Interactive column/row parallel patterns.
    [https://mbrenndoerfer.com/writing/tensor-parallelism-column-row-megatron-communication-patterns](https://mbrenndoerfer.com/writing/tensor-parallelism-column-row-megatron-communication-patterns)

14. **Meta Engineering Blog (2025)** — "Scaling LLM Inference: Innovations in Tensor Parallelism, Context Parallelism, and Expert Parallelism."
    [https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)

15. **Gibiansky, A. (2017)** — "Bringing HPC Techniques to Deep Learning." Baidu Research.
    [https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce)

16. **Horovod Concepts** — Collective operation descriptions for distributed deep learning.
    [https://horovod.readthedocs.io/en/stable/concepts_include.html](https://horovod.readthedocs.io/en/stable/concepts_include.html)

17. **Flux (2024)** — "Flux: Fast Software-based Communication Overlap on GPUs through Kernel Fusion." *arXiv:2406.06858*.
    [https://arxiv.org/abs/2406.06858](https://arxiv.org/abs/2406.06858)

18. **FlexLink (2025)** — "FlexLink: Boosting your NVLink Bandwidth by 27% without accuracy concern." *arXiv:2510.15882*.
    [https://arxiv.org/abs/2510.15882](https://arxiv.org/abs/2510.15882)

19. **ParallelKittens (2025)** — "ParallelKittens: A minimal CUDA framework for multi-GPU kernel design." Stanford HAI.
    [https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)

20. **NVRAR (2025)** — "LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication." *arXiv:2511.09557*.
    [https://arxiv.org/abs/2511.09557](https://arxiv.org/abs/2511.09557)

21. **NCCL 2.23 Blog** — "New Scaling Algorithm and Initialization with NVIDIA NCCL 2.23."
    [https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)

22. **NCCL 2.28 Blog** — "Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28."
    [https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)

23. **NIXL (2025)** — NVIDIA Interconnect eXchange Library for disaggregated serving KV cache transfer.
    [https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1/](https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1/)

24. **TensorRT-LLM MultiShot Blog (2024)** — "3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot."
    [https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/)

25. **CUDA IPC Documentation** — "Interprocess Communication." CUDA Programming Guide.
    [https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)

26. **TensorRT-LLM Source Code** — Collective communication implementations.
    [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
    - `cpp/tensorrt_llm/thop/allreduceOp.cpp` — C++ AllReduce strategy dispatch and topology detection
    - `cpp/tensorrt_llm/thop/allgatherOp.cpp` — C++ AllGather (NCCL / broadcast)
    - `cpp/tensorrt_llm/thop/reducescatterOp.cpp` — C++ ReduceScatter
    - `cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu` — Fused CUDA kernels
    - `cpp/tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.cu` — FP8 low-precision AllReduce for PCIe
    - `cpp/tensorrt_llm/kernels/cutlass_kernels/allreduce_gemm/` — GEMM+AllReduce fusion kernels (SM100+)
    - `tensorrt_llm/_torch/distributed/ops.py` — Python collective ops (AllReduce, AllGather, ReduceScatter, All-to-All)
    - `tensorrt_llm/_torch/modules/linear.py` — GEMM+AllReduce fusion gating and dispatch
    - `tensorrt_llm/_torch/modules/fused_moe/communication/` — MoE communication strategies
    - `tensorrt_llm/_torch/disaggregation/` — Disaggregated serving (NIXL, UCX, native transfer)
    - `tensorrt_llm/_torch/models/modeling_nemotron_h.py` — Nemotron-H model implementation
    - `tensorrt_llm/_ipc_utils.py` — CUDA IPC memory management and topology guards
    - `cpp/tensorrt_llm/common/customAllReduceUtils.h` — Lookup tables and utilities
