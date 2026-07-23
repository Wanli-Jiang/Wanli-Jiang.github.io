---
title: "Removing the Legacy TensorRT Backend from TensorRT-LLM: A Retrospective"
date: 2026-07-23
categories: [LLM, engineering]
tags: [TensorRT-LLM, refactoring, maintenance, CUDA, C++]
description: A retrospective on removing the legacy TensorRT engine backend from TensorRT-LLM through small, reviewable, always-compilable pull requests.
---

Over roughly four weeks, we removed the legacy **TensorRT engine backend** from TensorRT-LLM through a sequence of small, independently reviewable pull requests.

The effort spanned four phases and roughly twenty PRs. It removed about **200,000 lines of code** while preserving the behavior of the PyTorch and AutoDeploy execution paths.

This post records how we approached a deletion at that scale, what went well, what failed in surprising ways, and what lessons generalize to large refactors, decouplings, and removals.

---

## Why Remove It?

TensorRT-LLM originally used a **TensorRT engine backend** as its core execution path. Users converted a checkpoint, built a TensorRT engine with `trtllm-build`, and ran it through the `ModelRunner` family.

As the **PyTorch backend** became the default and **AutoDeploy** matured, the legacy backend became pure maintenance cost:

- **It was no longer the active path.** New model support, quantization formats, and parallelism strategies were landing on the PyTorch path.
- **It kept builds and images heavy.** Development images carried the full TensorRT SDK, wheels linked `libnvinfer`, and the C++ tree was full of `nvinfer1::` types.
- **It confused contributors.** Two executors, two decoder wirings, two packaging gates, and two test matrices made it easy for newcomers to choose the wrong path.

The goal was simple to state but hard to execute:

> Remove the legacy backend completely - examples, docs, tests, Python, C++, packaging, and CI images - without breaking PyTorch or AutoDeploy behavior.

The main constraint was stricter:

> Every PR had to compile and merge independently.

No "big bang" deletion. A 200k-line one-shot removal would be impossible to review, impossible to bisect, and too risky to land.

---

## Strategy: Outside-In, Four Phases

We removed the backend from the outside in:

```text
A. Outermost deletion
   examples / docs / tests / Triton C++ backend

B. Python backend
   gut in place, keep minimal wiring, then remove files

C. C++ tree
   remove nvinfer entirely; introduce owned DataType / Dims / ILogger

D. Trailing cleanup
   build flags / docker / Python relics / C++ relics
```

Three principles drove the order.

First, **delete leaves before roots**. Examples, docs, and tests are consumers of the backend. Removing those consumers first means the later backend removal leaves fewer dangling references.

Second, **remove Python before C++**. Once the Python entry path is gone, much of the C++ binding and wiring becomes dead code. That makes the C++ phase cleaner and easier to review.

Third, **reserve a dedicated cleanup phase**. After the bulk removal, a long tail remains: build flags, Docker install steps, lint bookkeeping, renamed files, and test residue. We isolated that into Phase D and split it by reviewer surface.

---

## Results

| Phase | Scope | PRs |
| --- | --- | --- |
| A | Examples, docs, tests, Triton C++ backend | #15763, #15767, #15810, #15907 |
| B | Python backend, hollowed out in place | #16213 |
| C | Remove `nvinfer` from C++ and packaging gate | #16369 |
| D.1 | Build flag and Docker/image cleanup | #16608 |
| D.2 | Residual tests, examples, CI plumbing | #16610 |
| D.3 | Python relics, docs, lint bookkeeping | TRTLLM-14474 |
| D.4 | C++ relics and orphaned modules | TRTLLM-14475 |

The two middle phases carried most of the weight:

- Phase B removed roughly **95,000 lines** from the Python backend.
- Phase C removed roughly **79,000 lines** from the C++ tree.

Every merged phase preserved the property we cared about most:

> The repository compiled at every step, and every step was reviewable on its own.

The cleanup also surfaced real bugs that had been masked by the legacy path, including an NVML initialization dependency and an OSS-mode CUTLASS compilation leak.

---

## What Worked

### Small, Always-Compilable PRs

No PR was allowed to depend on a future cleanup to compile. That made the sequence bisectable and kept review risk manageable.

Even the largest phases had a single concern:

- Phase B removed the Python execution path.
- Phase C removed the C++ `nvinfer` dependency.

Large diffs are not automatically bad. Large diffs become reviewable when the boundary is clear and the repository remains green after each step.

### Gut In Place Before Physical Deletion

For the Python backend, "gut in place" was safer than deleting every file immediately.

The first step hollowed out the backend while preserving enough wiring for imports and adjacent code to remain stable. Later PRs removed now-dead files.

That made the semantics of each diff clear:

```text
first: make the path unreachable / non-functional
then: remove consumers
then: remove dead files
then: remove packaging and CI residue
```

### Split Cleanup By Reviewer Surface

Phase D initially looked like one large cleanup branch. We split it into four PRs with disjoint file sets:

```text
infra / Docker / build flags
tests / examples / CI plumbing
Python / docs / lint bookkeeping
C++ orphaned files and modules
```

This was more effective than splitting by directory. The important question was not "where is this file?" but "who can review this safely?"

### Rebase Scans Protected Long-Lived Work

`main` kept moving while the deletion work was in flight. New TensorRT-typed code could be reintroduced upstream at any time.

After every rebase, we ran a scan suite that checked for reintroduced legacy symbols and files. That prevented "upstream added back the thing we are deleting" from reaching CI.

---

## Technical Lessons

### 1. Deleting An Import Chain Can Remove Load-Bearing Side Effects

After Phase B removed the Python backend, a family of NVFP4 models began running out of memory in CI. The first assumption was that this was unrelated.

It was related.

On `main`, `import tensorrt_llm` transitively initialized NVML through a module-level context deep in a profiler import chain. Removing that chain left NVML uninitialized. An unguarded NVML query in the multi-node memory layer then failed, NVLink communication probes failed, and the MoE communication factory silently chose a worse path that either hard-exited or consumed extra per-rank memory.

The branch did not directly change the MoE communication code. It removed an import chain that had been accidentally initializing the process.

We proved the issue with:

- a same-node A/B comparison between initialized and uninitialized NVML,
- an `nvmlInit` stack trace,
- a fixed rerun confirming that the correct communication backend was selected on all ranks.

The lesson:

> Before deleting an import chain, search for what it initializes, not just what it exports.

Examples include:

```text
NVML init
RTLD_GLOBAL loads
library preloads
process-level singletons
global allocator setup
plugin registration
```

### 2. Deleted C++ Targets May Have Load-Bearing PUBLIC Attributes

Two variants of the same problem appeared during the C++ phase.

The first full `google-tests` build failed to link several test binaries. Earlier builds had only produced wheel targets, so they did not expose the issue.

A deleted plugin target had been exporting PUBLIC includes and links that other targets relied on implicitly. One test got a quantization header transitively through the plugin. Another test needed a shared-library dependency but never declared it directly.

The second issue involved compile definitions. Mainline CI stayed green because tests linked the plugin, and the plugin leaked PUBLIC compile definitions into those tests. Removing the plugin stopped that leak, which activated an OSS-incompatible internal branch and broke a shared test fixture.

The lesson:

> After deleting a C++ target, reattach its PUBLIC includes, links, and defines to the real consumers explicitly.

Also:

> Build the full test suite, not just wheel targets.

A single compile error in a shared fixture can take down every scheduled C++ test that depends on it.

### 3. Type And Namespace Migrations Can Introduce Silent Ambiguity

Removing `nvinfer1` from C++ required introducing an in-house `DataType`, with values mirrored for serialization compatibility.

One test imported two namespaces at once and used an unqualified `DataType`. After the migration, that name became ambiguous.

A subtle trap: adding a same-scope alias like this does not necessarily solve ambiguity introduced by using directives:

```cpp
using DataType = tensorrt_llm::DataType;
```

An own-scope declaration does not simply hide all names introduced by using directives. The alias can become another candidate.

The safer fix is to narrow the import or fully qualify the name:

```cpp
tensorrt_llm::DataType dtype = ...;
```

### 4. Packaging Often Depends On Accidental Load-Time Bridges

Two dynamically loaded wrapper libraries had no direct dependency edge back to the core library, yet needed its symbols process-globally. The old code worked because another path loaded a library with `RTLD_GLOBAL`.

Removing the legacy path removed that accidental bridge.

The immediate fix was an explicit reload. The proper fix was to make the C++ layer self-sufficient and declare the dependency relationship directly.

The lesson:

> When removing a preload or dynamic link path, identify the accidental bridges it provided and verify the real runtime, not just `import`.

---

## Process Lessons

### Pin SHAs During History Surgery

Long-lived branches rebase onto a moving target. A rebase or soft reset against a moving `origin/main` can accidentally attach an old tree to a new parent and revert unrelated upstream commits.

The rule:

```text
pin the base by SHA
perform the history operation
verify the parent afterward
interdiff old and new heads against main's own range
```

Checking only the final tree is not enough. The parent matters.

### Resolve Generated Configs From The Authoritative Source

Generated artifacts such as lint baselines and config manifests can drift across tool versions if regenerated locally.

On conflict, rebuild deterministically from the upstream authoritative copy:

```text
drop entries whose files no longer exist
carry forward renames
avoid wholesale local regeneration unless required
```

### Split By Reviewer, Not Directory

The most effective cleanup split was ownership-based:

```text
infra reviewer
test/CI reviewer
Python/docs reviewer
C++ reviewer
```

Aligning PR boundaries to reviewer expertise reduced review latency more than a directory-based split would have.

### Write Down Environment Fragility

Long-running refactors accumulate environment knowledge:

```text
hook bootstrap steps
fetch quirks
dependency-version clashes
must-run setup commands before rerun
CI retry caveats
```

If that knowledge lives only in chat or memory, it will be rediscovered repeatedly. Capture it in project notes.

### Automate Reintroduction Detection

Any long-lived PR that removes a category of code should include a scan that runs after every rebase.

For this project, the scan blocked upstream reintroductions of:

```text
nvinfer references
legacy backend imports
deleted build flags
removed packaging gates
stale tests and examples
```

---

## Reusable Playbook

1. Phase the work outside-in: consumers, component, foundation, trailing cleanup.
2. Keep every PR independently compilable, single-concern, and bisectable.
3. Split cleanup by reviewer surface, not just by directory.
4. Audit import-chain side effects before deleting imports.
5. After deleting C++ targets, rebuild the full test suite.
6. Reattach PUBLIC includes, links, and defines to real consumers explicitly.
7. Ship a post-rebase scan suite to block upstream reintroductions.
8. Pin SHAs for history surgery and verify the parent afterward.
9. Resolve generated-config conflicts from authoritative upstream sources.
10. Record environment fragility in the project log.

---

## Summary

Through four phases and roughly twenty small, always-compilable PRs, we removed about **200,000 lines** of legacy TensorRT-backend code from TensorRT-LLM while preserving the PyTorch and AutoDeploy paths.

The deepest technical lesson was that large deletions do not remove isolated symbols. They remove import chains, transitive dependencies, linker behavior, compile definitions, runtime side effects, and accidental bridges.

The deepest process lesson was that long-lived deletion branches need active defense against a continuously moving `main`: pinned-SHA history operations, post-rebase interdiffs, and automated scans for reintroduced legacy code.

The end state is simpler: PyTorch and AutoDeploy are the active execution paths, development images no longer need to ship the TensorRT SDK for this legacy backend, and the C++ tree no longer carries `nvinfer` as a structural dependency.
