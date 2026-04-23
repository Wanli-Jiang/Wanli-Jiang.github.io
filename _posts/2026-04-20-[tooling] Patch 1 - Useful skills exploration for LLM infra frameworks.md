---
title: "Patch 1: Useful skills exploration for LLM infra frameworks"
date: 2026-04-20
categories: [LLM, tooling]
tags: [LLM, agent-skills]
description: A quick survey of the agent-facing skills and AGENTS.md guides shipped by vLLM, SGLang, FlashInfer, and TensorRT-LLM, so coding agents (and humans) can be productive in each codebase.
---

## Table of Contents

1. [Why this patch?](#1-why-this-patch)
2. [vLLM skills](#2-vllm-skills)
   - 2.1 [Where they live](#21-where-they-live)
   - 2.2 [The current skill catalog](#22-the-current-skill-catalog)
   - 2.3 [Concrete examples](#23-concrete-examples)
   - 2.4 [What I actually use](#24-what-i-actually-use)
3. [SGLang skills](#3-sglang-skills)
   - 3.1 [Where they live](#31-where-they-live)
   - 3.2 [The main-repo skill catalog](#32-the-main-repo-skill-catalog)
   - 3.3 [Concrete examples](#33-concrete-examples)
   - 3.4 [Patterns worth copying](#34-patterns-worth-copying)
   - 3.5 [The docs-repo AGENTS.md](#35-the-docs-repo-agentsmd)
   - 3.6 [Architecture crib sheet](#36-architecture-crib-sheet)
   - 3.7 [What I actually use](#37-what-i-actually-use)
4. [FlashInfer skills](#4-flashinfer-skills)
   - 4.1 [Where they live](#41-where-they-live)
   - 4.2 [The agent tools](#42-the-agent-tools)
   - 4.3 [Concrete examples](#43-concrete-examples)
   - 4.4 [What I actually use](#44-what-i-actually-use)
5. [TensorRT-LLM skills](#5-tensorrt-llm-skills)
   - 5.1 [Where they live](#51-where-they-live)
   - 5.2 [Rules you don't want the agent to forget](#52-rules-you-dont-want-the-agent-to-forget)
   - 5.3 [Architecture crib sheet](#53-architecture-crib-sheet)
   - 5.4 [The two local-only skills](#54-the-two-local-only-skills)
   - 5.5 [Concrete examples](#55-concrete-examples)
   - 5.6 [What I actually use](#56-what-i-actually-use)
6. [Adjacent skills worth stealing](#6-adjacent-skills-worth-stealing)
   - 6.1 [Training](#61-training)
   - 6.2 [Optimization and kernels](#62-optimization-and-kernels)
   - 6.3 [Inference runtimes](#63-inference-runtimes)
7. [Takeaways](#7-takeaways)
8. [References](#8-references)

---

## 1. Why this patch?

Over the past few months, every major LLM serving / kernel project has started shipping **agent-facing documentation** — either Anthropic-style `SKILL.md` bundles, an `AGENTS.md` at the repo root, or a dedicated Python `agents` module. The purpose is the same across all of them: give a coding agent (Claude Code, Cursor, etc.) the exact tribal knowledge a human maintainer would want it to have before touching the code or running the benchmarks.

This short "patch" post collects what I've found useful across the ecosystem. For each framework I list:

* **Where the skills live** — plugin repo, `.claude/skills/`, or Python module.
* **What they actually do** — a compact catalog so you can scan for the right tool.
* **Concrete examples** — for every skill, one realistic user prompt and what the agent produces.
* **What I actually use** — the two or three I reach for most often.

The four frameworks in the core survey (§2–§5) are **vLLM**, **SGLang**, **FlashInfer**, and **TensorRT-LLM**. §6 extends the tour into training, kernel optimization, and other inference runtimes.

---

## 2. vLLM skills

### 2.1 Where they live

vLLM has a dedicated plugin repository: [`vllm-project/vllm-skills`](https://github.com/vllm-project/vllm-skills) [1]. It follows the [anthropics/skills](https://github.com/anthropics/skills) template — every skill is a self-contained directory under `plugins/vllm-skills/skills/<skill-name>/` with a `SKILL.md` (YAML frontmatter + instructions) plus optional `scripts/`, `references/`, and `assets/`.

Install the whole bundle as a Claude Code plugin:

```bash
/plugin marketplace add vllm-project/vllm-skills
/plugin install vllm-skills@vllm-skills
```

Or copy a single skill into `~/.claude/skills/` (global) or `.claude/skills/` (project-scoped).

### 2.2 The current skill catalog

| Skill | What it does |
|-------|--------------|
| `vllm-deploy-simple` | Auto-detects hardware, installs vLLM, and brings up an OpenAI-compatible server with `vllm serve`. |
| `vllm-deploy-docker` | Deploys vLLM from the pre-built Docker image (or from source) with NVIDIA GPU support. |
| `vllm-deploy-k8s` | Deploys vLLM on Kubernetes with GPU requests, readiness/liveness probes, and an OpenAI-compatible endpoint. |
| `vllm-bench-random-synthetic` | Runs `vllm bench` against synthetic random prompts to measure throughput / TTFT / TPOT without touching HF datasets. |
| `vllm-bench-serve` | Benchmarks any OpenAI-compatible endpoint (vLLM or otherwise) with `vllm bench serve`. |
| `vllm-prefix-cache-bench` | Measures the efficiency of vLLM's automatic prefix caching using fixed prompts, a real dataset, or synthetic prefix/suffix patterns. |

### 2.3 Concrete examples

Each row below is one realistic user prompt (italic) followed by what the agent does when the skill is loaded.

**`vllm-deploy-simple`** — *"Serve Qwen2.5-7B-Instruct on this H100 box for local testing."*

Picks `--tensor-parallel-size 1 --gpu-memory-utilization 0.9 --dtype bfloat16`, launches `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`, polls `/health`, sanity-checks with a one-shot `POST /v1/chat/completions`, and returns the endpoint URL.

**`vllm-deploy-docker`** — *"Put this behind Docker so the build environment doesn't leak into prod."*

Pulls `vllm/vllm-openai:latest`, generates `docker run --gpus all --shm-size=16g -v $HF_HOME:/root/.cache/huggingface -p 8000:8000 vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct`, and confirms the container is healthy.

**`vllm-deploy-k8s`** — *"Deploy three replicas of Llama-3.1-8B on our GKE cluster with A100s."*

Writes a `Deployment` + `Service` manifest with `nvidia.com/gpu: 1`, a `nodeSelector: {cloud.google.com/gke-accelerator: nvidia-tesla-a100}`, readiness / liveness probes hitting `/health`, and runs `kubectl apply -f`.

**`vllm-bench-random-synthetic`** — *"What's peak throughput of this server before the network is in the picture?"*

Runs `vllm bench --model <model> --input-len 512 --output-len 128 --num-prompts 2000 --backend synchronous`, prints TTFT / TPOT / throughput, and flags whether the server is compute- or memory-bound.

**`vllm-bench-serve`** — *"Compare our staging endpoint to the current prod one under the same prompt mix."*

Runs `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --num-prompts 1000` against each URL and prints a side-by-side table of P50 / P99 latency and QPS.

**`vllm-prefix-cache-bench`** — *"Is prefix caching actually helping our RAG deployment?"*

Synthesizes a workload with a 2k-token shared system prompt + varying user questions, runs both with and without `--enable-prefix-caching`, and reports hit ratio + TTFT delta.

### 2.4 What I actually use

Two skills carry most of my daily weight:

1. **`vllm-deploy-simple`** — when I just need a known-good OpenAI endpoint in front of an HF checkpoint. The agent picks the right `--tensor-parallel-size`, `--gpu-memory-utilization`, and `--dtype` for the host instead of me re-googling my own shell history.
2. **`vllm-prefix-cache-bench`** — prefix caching is where most of the "is the deployment actually working?" questions live. This skill drives the bench tool with carefully designed prefix / suffix patterns so the cache hit ratio is interpretable, which is exactly what you want when triaging a regression.

---

## 3. SGLang skills

### 3.1 Where they live

SGLang has *two* separate skill stores, serving different audiences:

1. **Main repo**, [`sgl-project/sglang/.claude/skills/`](https://github.com/sgl-project/sglang/tree/main/.claude/skills) [2] — 10 in-repo `SKILL.md` bundles that the maintainers themselves reach for when adding kernels, debugging crashes, or bisecting CI regressions. This is the one you usually want.
2. **Docs repo**, [`sgl-project/sgl-docs/AGENTS.md`](https://github.com/sgl-project/sgl-docs/blob/main/AGENTS.md) [3] — a Mintlify docs-authoring skill with a strict source-of-truth hierarchy, covered in §3.5 below.

The main-repo bundle is the more interesting one because every skill is a **procedural playbook** with real command lines, real log excerpts, and real failure modes — exactly the tribal knowledge that usually lives in a senior engineer's head.

### 3.2 The main-repo skill catalog

| Skill | What it does |
|-------|--------------|
| `add-jit-kernel` | Step-by-step for adding a **lightweight JIT CUDA kernel** under `python/sglang/jit_kernel/`. Default path when the kernel doesn't depend on CUTLASS or another large C++ project. |
| `add-sgl-kernel` | Step-by-step for the **AOT / heavyweight** path under `sgl-kernel/` — `.cu` + `sgl_kernel_ops.h` declaration + `common_extension.cc` registration + `CMakeLists.txt` (alphabetical!) + Python wrapper + pytest + Triton benchmark. |
| `ci-workflow-guide` | The CI **infrastructure** layer: stage ordering (`stage-a/b/c`), fast-fail, gating, LPT auto-partition, slash commands, and how `pr-test.yml` / `pr-gate.yml` / `check-stage-health` fit together. |
| `write-sglang-test` | The CI **authoring** layer: always use `CustomTestCase` (not raw `unittest.TestCase`), defensive `tearDownClass`, `register_cpu_ci` / `register_cuda_ci`, test-placement rules, mock-over-real-server preference. |
| `debug-cuda-crash` | Five-level ladder of kernel-API logging via the `@debug_kernel_api` decorator: **L1** function names → **L3** inputs + metadata → **L5** tensor statistics (min/max/mean, NaN/Inf counts) → **L10** crash-safe `inputs.pt` dumps saved *before* execution. Integrates with `compute-sanitizer` + `cuda-gdb` and documents warp-specialized `printf` patterns. |
| `debug-distributed-hang` | Diagnose TP/PP/DP/EP hangs: py-spy / watchdog traces, on-demand CUDA coredumps (`CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1`), per-rank hashed-tensor event logs, and a binary-search method for finding the first-diverge point between ranks. |
| `generate-profile` | End-to-end recipe: launch `sglang serve`, poll `/health`, run `few_shot_gsm8k` as an accuracy gate (> 0.8), then `python -m sglang.test.send_one --profile` to emit a Chrome / Perfetto trace. |
| `sglang-auto-benchmark` | AI-driven tuning of server flags via `python -m sglang.auto_benchmark` — tiered search (tier 1/2/3), canonical autobench JSONL, real-traffic vs. synthetic datasets, SLA-bound QPS search, resume-on-SIGINT, and a hard rule to tune the base server before EAGLE speculation. |
| `sglang-bisect-ci-regression` | Triage scheduled `pr-test.yml` failures on `main`: `gh run list --event schedule` to find the pass → fail boundary, runner / GPU correlation tables (is it H200-specific?), and remote reproduction over SSH + Docker. |
| `sglang-torch-profiler-analysis` | Compact "three-table" triage of `torch.profiler` traces: kernel table, overlap-opportunity table, fuse-pattern table, all gated at ≥ 1% cumulative GPU-time. Supports single-trace and mapping + formal two-trace modes; cross-checks against the repo's fuse-overlap catalog before calling anything a new opportunity. |

### 3.3 Concrete examples

**`add-jit-kernel`** — *"Add a fast fused `silu * gate` elementwise kernel; no CUTLASS needed."*

Writes `python/sglang/jit_kernel/include/sgl_kernel/fused_silu.h`, a launcher using the skill's host-side abstractions (`host::RuntimeCheck`, `host::div_ceil`), a Python wrapper, pytest under `python/sglang/jit_kernel/tests/`, and a Triton-based benchmark under `benchmark/`. First-use compile, no wheel rebuild.

**`add-sgl-kernel`** — *"We need an FP8 GEMM epilogue that leans on CUTLASS."*

Creates `sgl-kernel/csrc/gemm/fp8_gemm_epilogue.cu`, adds the declaration to `include/sgl_kernel_ops.h`, registers the op in `csrc/common_extension.cc` with the `Tensor! out` in-place schema, appends the source to `CMakeLists.txt` (alphabetical!), writes a pytest comparing against `torch._scaled_mm`, and adds a Triton-style benchmark. `make build -j16` at the end.

**`ci-workflow-guide`** — *"Add a new `stage-c-test-8-gpu-h200` suite that only runs on PRs labelled `run-h200`."*

Edits `.github/workflows/pr-test.yml` to add the stage with the correct `needs:` gate, updates `test/run_suite.py` to recognize the suite name, and wires the label check into `pr-gate.yml` without breaking LPT auto-partition.

**`write-sglang-test`** — *"Add a unit test for the OpenAI request-translation middleware; it doesn't need a real server."*

Places the test at `test/registered/unit/test_openai_middleware.py`, uses `CustomTestCase` + `unittest.mock.patch`, registers it via `register_cpu_ci` so it lands in `stage-a-test-cpu`, and adds a defensive `tearDownClass` guarded by `hasattr`.

**`debug-cuda-crash`** — *"Qwen3-8B is crashing with `device-side assert triggered` on one of our requests. Find the input that kills it."*

Exports `SGLANG_KERNEL_API_LOGLEVEL=10`, `SGLANG_KERNEL_API_DUMP_DIR=/tmp/dumps`, and `SGLANG_KERNEL_API_DUMP_INCLUDE='sglang.custom_op.*'`, reproduces the crash, and returns `/tmp/dumps/.../inputs.pt` plus a `metadata.json` with `execution_status: "exception"` so you can replay the exact failing call offline.

**`debug-distributed-hang`** — *"A TP=8 run of our 70B model freezes every ~200 steps."*

Triggers `py-spy dump` on the stuck scheduler, enables on-demand CUDA coredump, sees `ncclDevKernel_AllGather_RING_LL` as the stuck kernel, adds per-rank logs that hash `extend_seq_lens` each step, diffs between ranks, and finds the first-diverge step — points at an EAGLE sampling non-determinism as root cause.

**`generate-profile`** — *"Get me a Chrome-tracing profile of Qwen3-8B on one H100 for the perf review."*

Runs `CUDA_VISIBLE_DEVICES=0 sglang serve --model-path Qwen/Qwen3-8B --port 30000 &`, polls `/health`, sanity-checks with `few_shot_gsm8k --num-q 20` (accuracy > 0.8), runs `python -m sglang.test.send_one --profile`, and hands back `/tmp/<ts>/<ts>-TP-0.trace.json.gz` with a `chrome://tracing` / Perfetto link.

**`sglang-auto-benchmark`** — *"Find me the best server config for a 1000 → 256 ISL/OSL chat workload at SLA `max_ttft_ms=800`, `max_tpot_ms=40`."*

Drops a YAML with `dataset.kind=random`, a tier-2 `search_space` over `{attention_backend, chunked_prefill_size, max_running_requests, cuda_graph_max_bs}`, `search.max_candidates=8`, runs `python -m sglang.auto_benchmark run`, and returns `summary.md` with the winning flags + the QPS it supports.

**`sglang-bisect-ci-regression`** — *"`test_lora_tp.py` has been red on scheduled runs for a week. Why?"*

Queries `gh run list --workflow=pr-test.yml --event schedule --branch main`, builds a runner / GPU correlation table, notices only `gpu-h200-worker-*` runners fail, walks commits in the pass → fail window filtered to `python/sglang/srt/lora/`, and ships a structured bisection report with short-term workaround + suspect PR.

**`sglang-torch-profiler-analysis`** — *"Here's a prefill trace from our DeepSeek-V3 run; anything obvious left on the table?"*

Runs `scripts/analyze_sglang_torch_profile.py --input /path/to/trace.json.gz`, returns the kernel table (top: `flash_attn_varlen_func` at 43%), the overlap table (`allreduce_rmsnorm_fusion` listed with `high` similarity — should already apply here), and the fuse-pattern table flagging a missed `silu_and_mul` fusion.

### 3.4 Patterns worth copying

A few structural patterns from this bundle are worth stealing for your own skill collections:

* **Skills cross-reference each other.** `write-sglang-test` sends you to `ci-workflow-guide` for pipeline questions; both send kernel work to `add-jit-kernel` / `add-sgl-kernel`. Cheap to implement, huge for keeping each skill focused.
* **Lowest-level rule first.** Every skill opens with 2–5 "core rules" or "rules of thumb" before any tutorial content. `add-sgl-kernel` leads with "Prefer `jit_kernel` first unless you need CUTLASS", which saves the agent from writing the wrong kind of kernel entirely.
* **Real log excerpts, not pseudo-output.** `debug-cuda-crash` pastes a real L3 log from `Qwen/Qwen3-0.6B` and a real L5 log from `FLUX.1-dev`. Agents condition much more reliably on concrete evidence than on "here's what it would look like".

### 3.5 The docs-repo AGENTS.md

The docs repo [`sgl-project/sgl-docs/AGENTS.md`](https://github.com/sgl-project/sgl-docs/blob/main/AGENTS.md) covers a different problem — writing Mintlify docs without hallucinating flags. The rules worth stealing regardless of whether you use Mintlify:

* **Source-of-truth hierarchy:** `docs.json` > current Sphinx docs at `docs.sglang.io` > upstream `sgl-project/sglang` code > cookbook.
* **Never guess flags, defaults, or behavior.** Verify against the upstream code, since CLI args change release-to-release.
* **Voice:** second person, sentence-case headings, prerequisites before commands.
* **Frontmatter:** every page needs `title`; internal links are root-relative without extensions; `mint broken-links` + `mint validate` before submit.

### 3.6 Architecture crib sheet

If you're pointing an agent at the SGLang Python runtime (`python/sglang/srt/`), the key entry points to put in its context are:

| Path | Role |
|------|------|
| `entrypoints/engine.py` | Coordinates tokenizer, scheduler, and detokenizer across processes. |
| `managers/scheduler.py` | Continuous batching, RadixAttention prefix caching, chunked prefill. |
| `mem_cache/` | Token-to-KV pool + the radix tree that backs prefix reuse. |
| `model_executor/model_runner.py` | Prefill / decode forward passes, CUDA graphs. |
| `layers/attention/` | Attention backends (including the FlashInfer backend, GQA/MQA). |

### 3.7 What I actually use

For daily perf work, two pieces carry most of the weight:

1. **`sglang-auto-benchmark`** when I need to tune a real deployment. The skill codifies the exact rules I otherwise forget (tune the base server *before* EAGLE, keep `mem_fraction_static` / `schedule_policy` out of the default search, cap `max_candidates` at 8, real user traffic beats `random` for production decisions) and the tiered search gives me a predictable time budget.
2. **`debug-cuda-crash`** when something is obviously broken. Level 3 catches 80% of shape / dtype / device-placement bugs on its own; level 10 with `SGLANG_KERNEL_API_DUMP_INCLUDE` scoped to the offending op is the fastest way to get a reproducible `inputs.pt` out of a crashing run.

For cross-framework comparisons I still fall back to raw `sglang.bench_serving`, since it drives `vllm`, `lmdeploy`, and `trt-llm` backends under the same CLI:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

---

## 4. FlashInfer skills

### 4.1 Where they live

FlashInfer is unusual: its "skills" are not `SKILL.md` files but a first-class **Python module** — [`flashinfer_bench.agents`](https://bench.flashinfer.ai/docs/api/python/rst/agents) [4]. Every function is designed for LLM tool-calling: JSON-serializable in and out, and convertible to an OpenAI / Anthropic function schema via `function_to_schema()` / `get_all_tool_schemas()`.

There is also the [`flashinfer-bench-starter-kit`](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) [5], which is the official template for the MLSys '26 kernel-generation contest and doubles as a reference workflow for any agent trying to author CUDA / Triton kernels against FlashInfer-Trace definitions.

### 4.2 The agent tools

| Tool | Purpose |
|------|---------|
| `flashinfer_bench_run_ncu(solution, workload, ...)` | Runs NVIDIA Nsight Compute on a solution and returns the text report. Supports section sets (`detailed`, `full`, etc.), kernel-name regex filtering, and timeouts. |
| `flashinfer_bench_list_ncu_options(...)` | Lists all valid NCU sets / sections so the agent can pick a legal `set=` / `sections=` parameter. |
| `flashinfer_bench_run_sanitizer(solution, workload, sanitizer_types=[...])` | Runs `compute-sanitizer` with any subset of `memcheck`, `racecheck`, `initcheck`, `synccheck`. |
| `pack_solution_from_files(path, spec, name, definition, author)` | Packs `.py` / `.cu` / `.cuh` / `.cpp` / `.h` from a directory into a `Solution` JSON object. |
| `extract_solution_to_files(solution, base_path)` | The inverse: materializes a `Solution` back onto disk with a `SOLUTION.md` metadata file. |
| `function_to_schema(func)` / `get_all_tool_schemas()` | Exposes the above as OpenAI / Anthropic-compatible function schemas — drop them straight into a tool-calling loop. |

The package also ships **FFI prompts** (`FFI_PROMPT_SIMPLE`, `FFI_PROMPT`) that document the [TVM FFI](https://tvm.apache.org/ffi/) binding API, which is the recommended way to expose CUDA kernels to Python in this ecosystem.

### 4.3 Concrete examples

**`pack_solution_from_files`** — *"Here's my `./attention_kernel_v3/` directory, turn it into a submittable solution."*

Calls `pack_solution_from_files("./attention_kernel_v3", spec=BuildSpec(language="cuda", target_hardware=["cuda"], entry_point="kernel.cu::sparse_attn"), name="attn_v3", definition="sparse_attention", author="me")`, and writes a `Solution` JSON you can feed straight to the sanitizer / NCU tools.

**`flashinfer_bench_run_sanitizer`** — *"Before I burn NCU cycles, is my kernel memory-safe?"*

Calls `flashinfer_bench_run_sanitizer(solution, workload, sanitizer_types=["memcheck", "racecheck"])`, parses the text report, and if it finds `Invalid __global__ write` reports the offending block / thread IDs and *skips* NCU until the kernel is clean.

**`flashinfer_bench_list_ncu_options`** — *"Which NCU set should I collect for a roofline analysis?"*

Calls the helper, parses `--list-sets` output, and picks `detailed` over `basic` / `full` based on the workload size. Passes the choice as `set=` to the next NCU run.

**`flashinfer_bench_run_ncu`** — *"Profile the kernel under the real contest workload."*

Runs `flashinfer_bench_run_ncu(solution, workload, set="detailed", page="details", kernel_name="sparse_attn.*", timeout=120)`, hands back the text report, and extracts SM occupancy, L2 hit rate, and DRAM bandwidth into a short summary so the next turn can decide whether the kernel is compute- or memory-bound.

**`extract_solution_to_files`** — *"I've got a submitted `solution.json` from last week; dump it back to a working tree."*

Calls `extract_solution_to_files(solution, "./work/attn_v3")`, which writes each `.cu` / `.py` source + a `SOLUTION.md` with the spec, so you can resume editing.

**`function_to_schema` / `get_all_tool_schemas`** — *"Expose these as native tool calls to my OpenAI agent."*

Runs `schemas = get_all_tool_schemas()` and passes them as `tools=schemas` to `client.chat.completions.create(...)`. The agent now has `flashinfer_bench_run_ncu` etc. as first-class function-call targets — no wrapper code needed.

**FFI prompts (`FFI_PROMPT_SIMPLE` / `FFI_PROMPT`)** — *"Wrap this CUDA kernel with TVM FFI so I can call it from Python."*

Injects `FFI_PROMPT` into the system message, then generates `binding.py` using `tvm.ffi.register_func` patterns that actually compile — instead of hallucinating a PyTorch-style `torch.utils.cpp_extension.load` call that fails silently on the contest runner.

### 4.4 What I actually use

For kernel work, the tight loop is:

1. `pack_solution_from_files(...)` → `Solution` JSON.
2. `flashinfer_bench_run_sanitizer(...)` with `memcheck` + `racecheck` — catches the obvious out-of-bounds / race issues before burning NCU time.
3. `flashinfer_bench_run_ncu(..., set="detailed", page="details")` — get the roofline numbers, pipe the text back to the agent, iterate.

The fact that all three return plain strings and accept plain JSON is what makes this usable as a tool-calling loop — there is no hidden Python state to reconcile between turns.

---

## 5. TensorRT-LLM skills

### 5.1 Where they live

TensorRT-LLM publishes an [`AGENTS.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/AGENTS.md) [6] at the repository root. It is opinionated in a very useful way, and it is paired with per-area developer guides — e.g. `tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md` and `.../fused_moe/MOE_DEVELOPER_GUIDE.md` — that `AGENTS.md` explicitly tells the agent to read before touching those files.

There is also a `CLAUDE.local.md` escape hatch: if that file exists next to `AGENTS.md`, the agent reads it and treats it as a developer-specific override of the shared guidance. That's how individual contributors ship per-workstation conventions (e.g. a custom `GH_CONFIG_DIR`) without polluting the shared doc.

### 5.2 Rules you don't want the agent to forget

Straight from the current `AGENTS.md`:

* Read and follow `CODING_GUIDELINES.md` for every C++ and Python change.
* NVIDIA copyright header on every new file; update the year on modified files.
* `git commit -s` for DCO; never attribute AI tools in the sign-off line — let `git` handle it.
* PR titles: `[JIRA/NVBUG/None][type] description`, e.g. `[TRTLLM-5516][perf] optimize cuda graph padding`.
* Set `LLM_MODELS_ROOT` before running anything under `tests/integration/`.
* Pre-commit hooks may rewrite files — if that happens, re-stage and commit again (don't `--amend` and don't skip the hooks).
* Python import style: `from package.subpackage import module`, not `from module import Class`.
* **TensorRT backend is legacy.** New features go to the PyTorch backend (`LLM(backend="pytorch")`) or AutoDeploy (`LLM(backend="_autodeploy")`). `TrtLlmArgs`, `trtllm-build`, `trtllm-refit`, and `convert_checkpoint.py` are bug-fix-only.

### 5.3 Architecture crib sheet

The common C++ core is shared between backends via Nanobind (scheduling, batch manager, KV cache, decoder, sampling). The agent-relevant entry points are:

| File | Role |
|------|------|
| `tensorrt_llm/llmapi/llm.py` | The `LLM(...)` user-facing API. |
| `tensorrt_llm/llmapi/llm_args.py` | Pydantic schema for `BaseLlmArgs` / `TorchLlmArgs` / `TrtLlmArgs`. |
| `tensorrt_llm/llmapi/llm_utils.py` | Model-specific default overrides (attention kernel, quant, spec-dec, cache). |
| `tensorrt_llm/_torch/pyexecutor/` | The default PyTorch executor path. |
| `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py` | AutoDeploy shim around `PyExecutor`. |
| `tensorrt_llm/executor/executor.py` | Backend-agnostic `GenerationExecutor`. |
| `tensorrt_llm/models/automodel.py` | Model auto-registry, resolved by the HF config `architectures` field. |

Request flow (worth pasting into an agent prompt verbatim):

```text
HuggingFace Model → LLM API → Executor (PyTorch / AutoDeploy / TensorRT)
    → Scheduler → Model Forward → Decoder → Sampling → Generated Tokens
```

### 5.4 The two local-only skills

Beyond `AGENTS.md`, the repo ships a couple of scoped skills under `.claude/skills/`:

* **`ci-failure-retrieval`** — step-by-step scripts to pull Jenkins test results via the CI API. Handy when a `/bot run` comes back red and you want the agent to fetch the logs without you copy-pasting URLs.
* **Reference configs** under `examples/configs/database/` — pareto-optimized serving configs across several models × GPUs × ISL/OSL × concurrency levels. `AGENTS.md` explicitly tells the agent to use these as starting points instead of hand-tuning.

### 5.5 Concrete examples

**`AGENTS.md` rules at PR time** — *"Open a PR that adds a new MoE expert-parallel path in `_torch/modules/fused_moe/`."*

Reads `MOE_DEVELOPER_GUIDE.md` first, stamps each new file with the NVIDIA copyright header, commits with `git commit -s`, formats the PR title as `[TRTLLM-12345][feat] add EP path to fused MoE`, and ignores any urge to `--amend` after pre-commit rewrites files (re-stages and commits again instead).

**`ci-failure-retrieval`** — *"The last `/bot run` on my PR is red, pull the actual failure."*

Follows the skill's Jenkins-API recipe, finds the failing stage (`DGX_H100-4_GPUs-PyTorch-1`), pulls the raw `stderr` slice containing the first `FAILED`, and quotes the 30 lines of context back so you can decide whether to `/bot run --disable-fail-fast` or fix locally.

**Reference configs** — *"Serve Llama-4-Maverick on 8×H200 with a reasonable starting config."*

Copies `examples/configs/database/llama-4-maverick-17b-128e-instruct-fp8.yaml`, checks ISL / OSL / concurrency match the target workload, and runs `trtllm-serve --port 8000 --config <file>` — instead of hand-picking `max_batch_size`, `kv_cache_free_gpu_memory_fraction`, and `moe_backend` from scratch.

**`test_to_stage_mapping`** — *"Which CI stage runs `test_llm_args::test_kv_cache_config`?"*

Runs `python scripts/test_to_stage_mapping.py --tests "test_kv_cache_config"`, gets the stage name (e.g. `A10-PyTorch-1`), and from there knows which runner / GPU the test lands on — so you know before the fact whether the PR needs `--extra-stage` or not.

### 5.6 What I actually use

For a contribution that touches the PyTorch backend, the loop looks like:

```bash
# 1) Unit test for the specific file I'm changing
pytest tests/unittest/llmapi/test_llm_args.py -v

# 2) Integration test once the unit test passes
LLM_MODELS_ROOT=/path/to/models pytest tests/integration/defs/...

# 3) Benchmark against a reference config
trtllm-bench --model throughput --dataset ...
trtllm-serve --port 8000 --config examples/configs/database/<model>.yaml
```

The `AGENTS.md` anti-patterns section has saved me more than once — in particular the "protected APIs" rule (changes to `LLM` API signatures will fail `tests/unittest/api_stability`) and the "one concern per PR" rule.

---

## 6. Adjacent skills worth stealing

The four serving / kernel frameworks above are my daily drivers, but the `SKILL.md` / `AGENTS.md` wave has reached the rest of the LLM lifecycle too. This section is the short version — "what else is out there and when I reach for it" — grouped by **training**, **optimization / kernels**, and **inference runtimes**.

### 6.1 Training

#### 6.1.1 `huggingface/skills` — the big tent

[`huggingface/skills`](https://github.com/huggingface/skills) [7] is the broadest skill bundle I know of — an `agents/AGENTS.md` registry at the root plus 11 individual `SKILL.md` directories. The ones most relevant to LLM work:

| Skill | What it does |
|-------|--------------|
| `hf-cli` | Drives the new `hf` CLI: auth, cache, repos, collections, HF Jobs, Inference Endpoints, buckets. Replaces the deprecated `huggingface-cli`. |
| `huggingface-llm-trainer` | SFT / DPO / GRPO / reward modeling via **TRL** or **Unsloth**, running on HF Jobs cloud GPUs. Handles dataset prep, hardware selection, cost estimation, Trackio monitoring, and GGUF conversion. |
| `huggingface-vision-trainer` | Object detection (D-FINE, RT-DETR v2, DETR, YOLOS), classification (timm, ViT / DINOv3), and SAM / SAM2 segmentation on HF Jobs. |
| `huggingface-community-evals` | Local evals with `inspect-ai` and `lighteval`; picks between vLLM / Transformers / accelerate backends. |
| `huggingface-trackio` | Experiment tracking + alerting for training runs, syncs to HF Spaces. |
| `huggingface-datasets` | Dataset Viewer API workflows (pagination, search, filters, parquet URLs). |
| `huggingface-tool-builder` | Generates reusable scripts that compose HF API calls. |
| `huggingface-paper-publisher` / `huggingface-papers` | Publish / look up papers on HF with linked models, datasets, and spaces. |
| `transformers-js` | Run Transformers models in JS / TS (browser + Node / Bun / Deno, WebGPU / WASM). |

**The pattern to copy:** an `agents/AGENTS.md` whose entire job is to *describe* the other skills ("if the user mentions X, read `skills/X/SKILL.md`"), so the agent doesn't have to scan the whole repo to find the right skill.

**Examples:**

**`hf-cli`** — *"Download the latest Qwen3 weights into our shared HF cache."*

Runs `hf auth login`, then `hf download Qwen/Qwen3-8B --local-dir $HF_HOME/Qwen3-8B`, confirms the hash, and prints the path.

**`huggingface-llm-trainer`** — *"SFT Llama-3.1-8B on our `internal/customer-support` dataset on an H100; make a GGUF when done."*

Writes a TRL Jobs config (PEP 723 UV script), picks the H100×1 hardware tier with a cost estimate, wires up Trackio, launches `hf jobs run`, and once training finishes runs the GGUF conversion (`Q4_K_M` by default) and pushes both artifacts to the Hub.

**`huggingface-vision-trainer`** — *"Fine-tune RT-DETR v2 on our COCO-formatted parking-lot dataset."*

Validates the annotations, configures Albumentations augmentation, picks an L4 GPU tier, launches a Jobs run, monitors mAP / mAR on Trackio, and pushes the trained model card with an eval table.

**`huggingface-community-evals`** — *"Evaluate our fine-tuned model on GSM8K and MMLU locally."*

Picks the `vllm` backend for speed, writes an `inspect-ai` config with both tasks, runs it against the checkpoint on the local GPU, and produces a markdown report with per-task scores.

**`huggingface-trackio`** — *"Alert me if training loss starts diverging."*

Adds a Trackio alert with a webhook to the team Slack, triggers on `loss > 10` or `grad_norm > 100`, and confirms the alert fires against a seeded failure run.

**`huggingface-datasets`** — *"Find all rows in `HuggingFaceH4/ultrachat_200k` where the system prompt mentions `coding`."*

Uses the Dataset Viewer API with `filter=contains(system, 'coding')`, paginates through results, and writes a local parquet with the filtered subset.

**`huggingface-tool-builder`** — *"I need to re-run this model-card update every week."*

Writes a reusable `update_model_card.py` UV script that reads the latest eval results from Trackio, renders the markdown table, and pushes via `HfApi` — parameterized so it runs standalone next week without re-prompting.

**`huggingface-paper-publisher`** — *"Publish our arxiv paper to HF and link the model + dataset."*

Creates the paper page, claims authorship, adds `models` and `datasets` backlinks, and generates the announcement markdown.

**`transformers-js`** — *"Run DistilBERT sentiment analysis in our React app with no backend."*

Scaffolds a `@xenova/transformers` snippet using WebGPU, pins a known-good quantized model, and adds a warm-up call to hide first-request latency.

#### 6.1.2 NVIDIA NeMo — `AGENTS.md` + skill directory

Two NeMo repos have adopted the pattern:

**`NVIDIA-NeMo/Automodel`** [8] ships a top-level `AGENTS.md` with hard rules (ruff line-length 120, `uv` only — no `pip install`, `safe_import()` for optional deps, Google-style docstrings, NVIDIA copyright header) plus six focused skills under `skills/`:

| # | Skill | Purpose |
|---|-------|---------|
| 1 | `model-onboarding` | Onboard a new LLM / VLM / OMNI / MoE / dLLM / text-to-image / text-to-video family. |
| 2 | `developer-guide` | Environment setup and day-to-day dev workflow. |
| 3 | `recipe-development` | Create / modify training and eval recipes. |
| 4 | `distributed-training` | FSDP2, HSDP, pipeline parallelism, context parallelism. |
| 5 | `parity-testing` | Numerical correctness vs. reference implementations. |
| 6 | `launcher-config` | Slurm and SkyPilot job submission. |

The `AGENTS.md` also documents the non-obvious invariants that the test suite won't catch: models must register in `MODEL_ARCH_MAPPING`, MoE models need `MoEFSDPSyncMixin` for correct expert gradient sync under FSDP2, fused QKV / GateUp projections use **interleaved layout** so TP splits evenly across heads / experts, and `BackendConfig` is the only legal place to pick kernel implementations.

**Examples:**

**`model-onboarding`** — *"Add support for the new DeepSeek-V3.2 architecture."*

Creates `components/models/deepseek_v3_2/{model.py, state_dict_adapter.py, config.py}`, inherits from `PreTrainedModel` + `HFCheckpointingMixin` (+ `MoEFSDPSyncMixin` since it's MoE), registers the arch string in `MODEL_ARCH_MAPPING`, declares `supports_fp8=True` / `supports_moe=True` in `capabilities.py`, writes an HF ↔ NeMo weight-key mapping, and opens a PR.

**`developer-guide`** — *"Set up a fresh dev env on this machine."*

Uses `uv sync` (not `pip install`), verifies Python 3.10+ / PyTorch 2.6+, runs `ruff format . && ruff check --fix .` on a smoke test, and confirms `pytest -q` passes before you touch real code.

**`recipe-development`** — *"Write a Llama-3.1-70B SFT recipe for our 8×H100 box."*

Scaffolds `recipes/llm/llama31_70b_sft.yaml` with `_target_: ...LlamaForCausalLM`, points `build_model` / `build_optimizer` / `build_dataloader` / `build_trainer` at the right components, and leaves a TODO for the data path so you can plug in your own parquet.

**`distributed-training`** — *"Train this 70B model with FSDP2 + 8-way TP + context parallelism."*

Builds the device mesh via `infrastructure.py`, sets the FSDP2 sharding policy, verifies the MoE gradient-sync mixin is on, and drops a sanity-check script that runs one forward / backward on 2 nodes before scaling up.

**`parity-testing`** — *"Prove our port matches reference HF logits to 1e-3."*

Writes a test that loads the HF checkpoint and the ported NeMo model with the same seeds, runs a single forward on a fixed input, asserts `torch.allclose(logits_hf, logits_nemo, atol=1e-3, rtol=1e-3)`, and flags any divergence back to the weight mapping.

**`launcher-config`** — *"Submit this recipe to our Slurm cluster."*

Writes a Slurm submission script with the right `#SBATCH --gpus-per-node=8 --nodes=4 --time=24:00:00`, container entrypoint, and env propagation for `WANDB_API_KEY` / `HF_TOKEN`. Optionally hands off to SkyPilot if the cluster spec changes.

**`NVIDIA-NeMo/Megatron-Bridge`** [9] added a [`skills/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) directory in March 2026 with structured guides aimed at Cursor / Claude Code / Codex for adding model support, setting up dev environments, and tuning performance. There's also a dedicated [Agent Skills Reference](https://docs.nvidia.com/nemo/megatron-bridge/latest/skills-index.html) page in the docs.

**Example** — *"Convert our HF Llama-3.2-1B checkpoint to Megatron-Core and validate the round-trip."*

Reads the "add model support" skill, runs `AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")`, materializes the Megatron model, exports back to HF, and uses the skill's verification recipe to compare a fixed forward between the original HF checkpoint and the round-tripped one.

#### 6.1.3 Axolotl and the "research skill pack" pattern

[`axolotl-ai-cloud/axolotl`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/AGENTS.md) [10] ships its own `AGENTS.md` documenting the YAML config patterns, FSDP setup, context parallelism, and compressed model saving conventions the framework expects. There's also a community-maintained `axolotl` skill (via `orchestra-research/ai-research-skills`) that wraps the same conventions in `SKILL.md` form for agents that prefer per-skill directories over a monolithic `AGENTS.md`.

The whole `orchestra-research/ai-research-skills` pack [11] has `awq`, `gptq`, and `axolotl` skills side-by-side — a nice model for "one repo, many small research-task skills" if you want to build an in-house version.

**Example** — *"Fine-tune Mistral-7B with QLoRA + ORPO on our preference dataset via Axolotl."*

Writes an Axolotl YAML with `rl: orpo`, `adapter: qlora`, correct FSDP settings for a 4×A100 node, and the compressed `save_safetensors` flag per the skill's guidance. Launches with `accelerate launch -m axolotl.cli.train config.yml` and surfaces the final adapter path.

### 6.2 Optimization and kernels

#### 6.2.1 `agent-gpu-skills` — abstraction-level-organized kernel work

[`slowlyC/agent-gpu-skills`](https://github.com/slowlyC/agent-gpu-skills) [12] is organized not by framework but by **abstraction level**, which matches how you actually think when writing a kernel:

| Skill | Layer |
|-------|-------|
| `cuda-skill` | PTX / CUDA C++ |
| `cutlass-skill` | CUTLASS / CuTe DSL |
| `triton-skill` | Triton (Python DSL) |
| `sglang-skill` | LLM serving layer on top |

Each skill ships doc libraries, source-code references, and install scripts so the agent can pull up the right examples for the right layer.

**Examples:**

**`cuda-skill`** — *"Write a raw CUDA kernel for in-place layer-norm on FP16."*

References the bundled PTX / CUDA C++ snippets, produces a `layernorm.cu` using warp-level reductions, sets `blockDim.x = 256`, and adds the launch configuration + a `cudaGetLastError()` check.

**`cutlass-skill`** — *"Build a BF16 GEMM using CUTLASS 3.x CuTe DSL for H100."*

Picks `cutlass::gemm::collective::CollectiveMma` with tile shape `<128, 128, 64>`, pulls in `cutlass::arch::Sm90`, configures the epilogue, and wires the host runner.

**`triton-skill`** — *"Write a fused RMSNorm + residual-add kernel in Triton."*

Uses `@triton.jit` with correct `BLOCK_SIZE` autotuning, writes both a correctness test vs. `torch` reference and a `triton.testing.perf_report` benchmark.

**`sglang-skill`** — *"Integrate this new Triton kernel into SGLang's serving path."*

Reads the install script, places the kernel under `python/sglang/jit_kernel/`, registers it via `register_custom_op`, and adds the call site in the model's forward.

#### 6.2.2 Kernel-generation contests as skills

The [**FlashInfer MLSys '26 starter kit**](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) covered in §4 doubles as a "how to author CUDA / Triton kernels with a coding agent" skill, because `scripts/run_local.py`, `scripts/pack_solution.py`, `flashinfer_bench_run_ncu`, and `flashinfer_bench_run_sanitizer` are all designed to be called by an agent loop. This is the most direct answer to "what should a kernel-writing agent actually do between turns?".

There's also a community-curated [`cutlass-triton`](https://playbooks.com/skills/a5c-ai/babysitter/cutlass-triton) skill [13] focused on GEMM configuration generation, epilogue ops, tile / warp tuning, and benchmarking against cuBLAS — useful as a template if you want to carve off a narrower "just GEMMs" sub-skill.

**Example** — *"Generate a tuned BF16 GEMM for `M=8192, N=8192, K=8192` on H100 and compare to cuBLAS."*

Emits a CUTLASS configuration with tile `<128, 256, 64>`, 4-stage pipelining, and a fused-bias epilogue; runs it next to `torch.matmul`; prints TFLOPS for each and the % of cuBLAS speed-of-light.

#### 6.2.3 Quantization skills

Post-training quantization is a natural fit for skills because each method (AWQ, GPTQ, FP8, SmoothQuant, …) has a fixed recipe. The `orchestra-research/ai-research-skills` pack [11] has `awq` and `gptq` entries that wrap the `llm-compressor` and `auto-gptq` / `autoawq` flows (calibration data, group sizes, kernel selection, accuracy checks). The one I keep an eye on is `llm-compressor`, since it's the upstream quant pipeline that now lives under `docs.vllm.ai/projects/llm-compressor/`.

**Examples:**

**`awq`** — *"AWQ-4bit this Llama-3.1-70B so it fits on 2×A100."*

Prepares a 128-sample calibration set from `pile-val`, runs `AutoAWQForCausalLM.from_pretrained(...).quantize(...)` with `group_size=128, zero_point=True`, evaluates PPL on WikiText to confirm < 1% degradation, and writes out the AWQ checkpoint.

**`gptq`** — *"GPTQ-4bit our Qwen2.5-32B for use with the ExLlamaV2 backend."*

Runs `auto-gptq` with `bits=4, group_size=128, desc_act=True`, validates perplexity, saves in the `exllamav2` layout, and prints the command to launch it under vLLM.

### 6.3 Inference runtimes

#### 6.3.1 `llama.cpp` — first-class skills + AGENTS.md + MCP

`ggml-org/llama.cpp` landed [Skills, AGENTS.md, and MCP support](https://github.com/ggml-org/llama.cpp/commit/c42a7477f44b47f0ab9abc8ce6e6f11743a859d7) [14] in its agent server in early 2026. What's nice about their implementation is that it defines a clear loading model that's worth imitating:

* `SKILL.md` directories are **discovered per-session** based on the working directory and injected into the agent configuration.
* `AGENTS.md` files are **also discovered per-session** and injected as prompt sections — so per-project guidance composes with global skills.
* MCP servers are initialized at startup (Unix only) and tools are registered globally across all sessions.

Session-level knobs: `enable_skills: bool`, `enable_agents_md: bool`, `skills_paths: list`. If you've ever wondered "how should an inference server actually surface agent skills?", this is a very clean reference.

**Example** — *"Start the llama.cpp agent server from our project repo and load its AGENTS.md."*

`llama-agent --enable-skills --enable-agents-md --skills-paths ~/.llama/skills:./skills --model ./llama-3.1-8b-q4.gguf`. When the session opens in the project's `cwd`, the local `AGENTS.md` is injected as a prompt section and the local `skills/` is concatenated onto the global skills path — so per-project rules automatically win over global defaults.

#### 6.3.2 Meta-skills: skill-creator and the agentskills.io spec

Two bits of infrastructure I come back to whenever I write a new skill:

1. **[`openai/skills/.system/skill-creator/SKILL.md`](https://github.com/openai/skills/blob/main/skills/.system/skill-creator/SKILL.md)** [15] — a skill whose job is to create other skills. Useful both as a template and as a live tool when you want the agent to generate a new skill on demand.
2. **The [agentskills.io](https://agentskills.io/) spec** — the shared format for `SKILL.md` YAML frontmatter (`name`, `description`, optional `license`, `metadata`, etc.) that vLLM, SGLang, FlashInfer-Bench, llama.cpp, and HuggingFace all target. Writing to this spec means your skills load unchanged across Claude Code, Cursor, Codex, Gemini CLI, and the llama.cpp agent server.

**Examples:**

**`skill-creator`** — *"Create a new skill called `trtllm-kv-cache-debug` that runs our custom KV-cache dump script and summarizes the output."*

Scaffolds `skills/trtllm-kv-cache-debug/SKILL.md` with agentskills.io-compliant frontmatter (`name: trtllm-kv-cache-debug`, `description: "..."`), a `scripts/dump_kv_cache.py`, an example invocation, and a `references/` subdir with pointers to TensorRT-LLM's KV-cache docs. Adds a `# Rules` section before the tutorial body per the skill-creator conventions.

**agentskills.io spec compliance** — *"Validate that our 12 in-house skills will load in Cursor, Claude Code, and Codex without edits."*

Runs a linter against each `SKILL.md`: checks `name` is slug-cased, `description` fits the 1024-char limit, frontmatter is strict YAML (no tabs), the body leads with rules, and the top-level `/skills/<name>/` directory matches `name`. Fails any skill that doesn't round-trip across runtimes.

---

## 7. Takeaways

1. **Skill shape is converging on three formats:** a `SKILL.md` plugin bundle (vLLM, HuggingFace, NeMo Automodel, SGLang), an `AGENTS.md` at repo root (SGLang docs, TensorRT-LLM, Axolotl, NeMo Automodel), and a Python `agents` module exposed as tool schemas (FlashInfer). They aren't mutually exclusive — TensorRT-LLM and SGLang both use `.claude/skills/` + an `AGENTS.md`; HuggingFace uses `AGENTS.md` as a *registry* for many `SKILL.md`s; llama.cpp loads both per-session.
2. **The highest-ROI skills are the boring ones:** "deploy this server", "benchmark that endpoint", "run the sanitizer, then NCU", "fine-tune with TRL on HF Jobs". They turn tacit ops knowledge into reproducible automation.
3. **Point the agent at the right crib sheet**, not the whole repo. The per-area developer guides that `AGENTS.md` links to (attention, MoE, distributed training, parity testing) are what actually prevent the agent from breaking invariants the test suite doesn't cover.
4. **Always verify flags against the upstream code.** This is the single rule I copy-paste into every new agent config, and it comes straight from SGLang's docs-repo `AGENTS.md`.
5. **Organize kernel skills by abstraction level, not by framework.** `agent-gpu-skills` (CUDA → CUTLASS → Triton → serving) matches how the work actually decomposes.
6. **Target the [agentskills.io](https://agentskills.io/) spec** so a single skill works across Claude Code, Cursor, Codex, Gemini CLI, and the llama.cpp agent server.

I'll keep this as a living patch — when a new skill lands I'll update the tables above.

---

## 8. References

1. **vLLM Skills:** [`vllm-project/vllm-skills`](https://github.com/vllm-project/vllm-skills).
2. **SGLang main-repo skills:** [`sgl-project/sglang/.claude/skills/`](https://github.com/sgl-project/sglang/tree/main/.claude/skills).
3. **SGLang docs AGENTS.md:** [`sgl-project/sgl-docs/AGENTS.md`](https://github.com/sgl-project/sgl-docs/blob/main/AGENTS.md).
4. **FlashInfer-Bench agents module:** [`flashinfer_bench.agents` API reference](https://bench.flashinfer.ai/docs/api/python/rst/agents).
5. **FlashInfer-Bench starter kit:** [`flashinfer-ai/flashinfer-bench-starter-kit`](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit).
6. **TensorRT-LLM AGENTS.md:** [`NVIDIA/TensorRT-LLM/AGENTS.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/AGENTS.md).
7. **HuggingFace Skills:** [`huggingface/skills`](https://github.com/huggingface/skills).
8. **NeMo AutoModel AGENTS.md:** [`NVIDIA-NeMo/Automodel/AGENTS.md`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/AGENTS.md).
9. **NeMo Megatron-Bridge skills:** [`NVIDIA-NeMo/Megatron-Bridge/skills/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) and the [Agent Skills Reference](https://docs.nvidia.com/nemo/megatron-bridge/latest/skills-index.html).
10. **Axolotl AGENTS.md:** [`axolotl-ai-cloud/axolotl/AGENTS.md`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/AGENTS.md).
11. **Research skills (AWQ / GPTQ / Axolotl):** [`orchestra-research/ai-research-skills`](https://github.com/orchestra-research/ai-research-skills).
12. **Agent GPU Skills:** [`slowlyC/agent-gpu-skills`](https://github.com/slowlyC/agent-gpu-skills).
13. **CUTLASS-Triton skill:** [`a5c-ai/babysitter` — cutlass-triton](https://playbooks.com/skills/a5c-ai/babysitter/cutlass-triton).
14. **llama.cpp agent server:** [feat(agent-server): add Skills, AGENTS.md, and MCP support](https://github.com/ggml-org/llama.cpp/commit/c42a7477f44b47f0ab9abc8ce6e6f11743a859d7).
15. **OpenAI skill-creator:** [`openai/skills/.system/skill-creator/SKILL.md`](https://github.com/openai/skills/blob/main/skills/.system/skill-creator/SKILL.md).
16. **Anthropic skills template:** [`anthropics/skills`](https://github.com/anthropics/skills).
17. **agentskills.io spec:** [https://agentskills.io/](https://agentskills.io/).
