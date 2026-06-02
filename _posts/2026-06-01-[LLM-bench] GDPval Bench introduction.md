---
title: GDPval Bench introduction
date: 2026-06-01
categories: [LLM, bench]
tags: [LLM, agents, evaluation]
description: An exploration of GDPval, OpenAI's benchmark for evaluating AI models on realistic, economically valuable knowledge-work tasks.
---

# 1. Why do we need GDPval?

## 1.1 From exam questions to real work

Many LLM benchmarks are still shaped like school exams: solve a math problem, answer a science question, write a short program, or pick the right multiple-choice option. These evaluations are useful because they are clean, repeatable, and easy to grade. However, they only capture a small slice of what people actually want from AI systems.

In real work, the output is rarely a single token or a short answer. A professional may need to write a legal memo, prepare a slide deck, analyze a spreadsheet, clean up a customer-support workflow, summarize clinical context, or turn messy reference files into a polished deliverable. The task is not just "know the answer"; it is "produce something useful enough that another professional would accept it."

This is the gap that **GDPval** tries to measure.

## 1.2 Economic value as the organizing principle

GDPval starts from a direct question: if AI models are becoming useful in the economy, can we measure their performance on tasks that resemble economically valuable work?

Instead of sampling tasks from academic subjects, GDPval samples from occupations in major U.S. GDP-contributing sectors. The benchmark focuses on digital knowledge work: work that can be performed on a computer and judged by professionals in the relevant field.

This makes GDPval interesting for two reasons:

* It tests whether models can produce real professional deliverables, not just correct text answers.
* It connects benchmark design to labor-market structure, using sectors, occupations, wages, and work activities as part of the sampling process.

## 1.3 Why this matters for agents

GDPval is also a useful benchmark for thinking about AI agents. Many tasks require reading reference files, deciding what matters, using tools, creating artifacts, checking formatting, and making tradeoffs between correctness and presentation.

That is much closer to how modern agentic systems are used. A model may need to write code, inspect generated files, revise a presentation, or reason over multimodal inputs before producing the final answer. GDPval therefore gives us a more realistic view of where models are already useful and where they still need human supervision.


# 2. What is GDPval?

## 2.1 Overview

GDPval is a benchmark introduced by OpenAI in the paper **"GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks"** [1]. The first version contains:

* **1,320 tasks** in the full benchmark.
* **220 tasks** in the open-source gold subset.
* **44 occupations**.
* **9 sectors** selected from major contributors to U.S. GDP.
* Tasks authored and reviewed by experienced professionals, averaging about **14 years of experience**.

Each occupation has 30 tasks in the full set and 5 tasks in the open gold subset. The public subset is available as the Hugging Face dataset `openai/gdpval` [2].

## 2.2 What makes GDPval different?

GDPval is not designed around short, fully deterministic answers. It is designed around professional work products. A task usually contains:

* **A request:** the instruction or work order given to the model.
* **Reference files:** supporting materials such as documents, spreadsheets, images, slides, audio, video, CAD-like files, or customer-support records.
* **A deliverable:** the final artifact expected from the model, such as a document, spreadsheet, diagram, slide deck, analysis, or written recommendation.

This changes the evaluation problem. The model has to understand context, produce an artifact in the correct format, and satisfy professional expectations. In many cases, the quality of the deliverable depends on judgment, aesthetics, structure, completeness, and correct use of the reference files.

## 2.3 The occupations and sectors

The benchmark covers sectors such as:

* Real estate and rental/leasing
* Manufacturing
* Professional, scientific, and technical services
* Government
* Health care and social assistance
* Finance and insurance
* Retail trade
* Wholesale trade
* Information

Within these sectors, GDPval chooses high-compensation, predominantly digital occupations. Examples include software developers, lawyers, accountants, financial managers, registered nurses, industrial engineers, editors, producers, and customer service representatives.

The important point is not that GDPval covers every job. It does not. Instead, it is an initial, structured attempt to sample realistic knowledge-work tasks from economically significant areas.

## 2.4 What does GDPval mean in Artificial Analysis?

Artificial Analysis uses the name **GDPval-AA** for its own independent evaluation built on OpenAI's GDPval dataset [5]. The "AA" suffix matters because it is not exactly the same reporting setup as the OpenAI paper.

In OpenAI's original GDPval paper, the key benchmark object is the dataset and expert-grading protocol: model deliverables are compared with expert human deliverables by professional graders. In Artificial Analysis, GDPval-AA is an **agentic leaderboard**: models are placed in an agent harness called **Stirrup**, given shell access and web browsing capability, and asked to solve the GDPval tasks by producing complete deliverables. The outputs are then compared head-to-head, and the pairwise results are aggregated into an **Elo rating**.

So when reading Artificial Analysis results, I interpret the metric as:

* **Dataset:** OpenAI GDPval tasks.
* **Execution mode:** agentic, with shell and web access through Stirrup.
* **Scoring format:** blind pairwise comparisons between model submissions.
* **Final leaderboard number:** an Elo score for relative agent performance.
* **Composite use:** GDPval-AA is one component of the Artificial Analysis Intelligence Index v4.0, alongside benchmarks such as Terminal-Bench Hard, IFBench, GPQA Diamond, Humanity's Last Exam, and others.

This distinction is important when comparing numbers. A GDPval-AA Elo score is not the same as OpenAI's "win rate vs. human expert deliverable." It is a relative leaderboard score under Artificial Analysis' harness and judging setup.


# 3. How GDPval tasks are built

## 3.1 Task sourcing

GDPval tasks are written by professionals from the relevant occupations. The paper describes a multi-step process:

1. Select sectors that contribute more than 5% of U.S. GDP.
2. Choose occupations within those sectors that are both economically significant and predominantly digital.
3. Recruit experienced professionals to create realistic tasks based on actual work patterns.
4. Map tasks to O*NET work activities to improve coverage and representativeness.
5. Review and revise tasks through multiple rounds of quality control.

This gives GDPval a different flavor from benchmarks that are created mainly by researchers or crowdworkers. The task is supposed to feel like something a professional might actually be asked to do.

## 3.2 Task anatomy

A typical GDPval task can be thought of as:

```text
Request + reference files -> model work process -> professional deliverable
```

For example, the benchmark may ask a model to:

* Draft a legal-style document from a set of case materials.
* Produce an engineering or operations analysis using reference diagrams.
* Create a customer-support response plan from a conversation transcript.
* Build a nursing care plan or administrative healthcare artifact from structured context.
* Prepare slides or spreadsheet outputs with specific formatting expectations.

The model is judged not only on factual correctness, but also on whether the output is useful, complete, well-structured, and faithful to the requested format.

## 3.3 Concrete examples

The public OpenAI blog gives a useful manufacturing example [4]. The model is asked to act as a manufacturing engineer reviewing a final testing step for a cable spooling truck. The current test requires two people to reel cable in and out of the truck. The manager asks for a jig or fixture so the test can be done by one person. The model receives an information document with cable reel drum dimensions and deliverable requirements. The expected output is not a short text answer; it is a preliminary concept design summarized as a PDF, using snapshots from a 3D design or equivalent visual concept work.

That task tests several capabilities at once:

* Understanding an industrial workflow and safety/labor constraint.
* Translating reference dimensions into a plausible fixture concept.
* Producing a professional design summary.
* Creating visuals that communicate the idea clearly.
* Exporting the final artifact in the requested format.

Other GDPval-style tasks include:

* **Legal work:** draft a legal-style memo or brief from provided case facts, constraints, and reference documents. The grade depends on issue spotting, legal reasoning, structure, and professional tone.
* **Finance work:** analyze spreadsheet data, prepare a forecast or investment-style summary, and deliver an Excel workbook or executive memo.
* **Healthcare administration:** produce a nursing care plan or operational healthcare document from patient or workflow context, where clinical accuracy and formatting both matter.
* **Customer support:** inspect a support conversation or policy context and produce a response plan, escalation summary, or customer-facing communication.
* **Retail or wholesale strategy:** turn sales, inventory, or market data into an executive presentation with charts and recommendations.
* **Information/media work:** edit, summarize, or package content into a publishable artifact, where style, relevance, and audience fit are part of the score.

The common pattern is: GDPval asks for **work product**, not just reasoning. A model can understand the prompt but still fail if the final PDF is corrupted, the spreadsheet formulas are wrong, the chart labels are unreadable, or the deliverable ignores a formatting instruction.

## 3.4 Multimodal and file-heavy work

One of GDPval's most important design choices is that tasks are file-heavy. The paper notes that gold-subset tasks can include many reference files, while the full set can include even more. The files may involve different modalities and formats.

This makes GDPval especially relevant for modern tool-using models. A text-only model that can reason well may still fail if it cannot inspect a spreadsheet, render a PDF, understand a chart, or verify that a PowerPoint slide actually looks right.


# 4. How GDPval is graded and scored

## 4.1 Human expert pairwise comparison

GDPval's primary grading method is **blind pairwise expert comparison**. A professional grader sees the task request, the reference files, and multiple deliverables. The grader then compares the outputs without being told which model or human produced them.

This is necessary because many GDPval outputs do not have a single exact answer. A legal memo can be more or less persuasive. A slide deck can be more or less clear. A spreadsheet can be technically correct but poorly formatted. A professional judgment is needed.

The headline metric is therefore based on whether a model's deliverable is preferred to, or judged comparable with, the human expert deliverable.

OpenAI maps each comparison to an ordinal score:

* `1.0`: model deliverable is preferred to the human deliverable.
* `0.5`: model deliverable is considered as good as the human deliverable.
* `0.0`: human deliverable is preferred.

Depending on the report, you may see **wins only** or **wins plus ties**. This is one reason GDPval numbers can look different across sources. A strict win-rate number counts only `1.0`; a parity-style number may count both `1.0` and `0.5`.

## 4.2 Automated grading as a proxy

OpenAI also provides automated grading support for the public gold subset [3]. The paper is careful about this: automated grading is useful for faster iteration, but it is not the same as expert human grading.

In the reported results, the automated grader reached about **66% agreement** with human expert graders, while human inter-rater agreement was about **71%**. That is close enough to be useful for research iteration, but still too noisy to treat as a perfect replacement for expert evaluation.

For serious claims, GDPval should still be interpreted through the lens of human expert preference.

The OpenAI Evals page later notes that GDPval rubrics and gold deliverables have been open-sourced, so researchers can now run their own grading workflows as well [3]. The recommended standard remains pairwise human expert preference, while LLM-based judging should be treated as a rough estimate.

## 4.3 Artificial Analysis Elo

Artificial Analysis reports GDPval-AA as an **Elo leaderboard** rather than a direct "percent win vs. human" number [5]. The basic idea is:

1. Run each model/agent on GDPval tasks in the same harness.
2. Collect final deliverables.
3. Compare two anonymized model submissions on the same task.
4. Ask an LLM judge to choose which submission is better.
5. Fit the resulting pairwise win/loss data to a rating model.
6. Publish an Elo score that reflects relative strength in that evaluation environment.

This is useful for ranking agentic systems, but it changes the interpretation. Elo answers "which model tends to beat which other model under this setup?" OpenAI's original win-rate metric answers "how often does this model beat or match the human expert deliverable?"

The judge is important. According to the Artificial Analysis Intelligence Benchmarking methodology, GDPval-AA uses **Gemini 3.1 Pro Preview** to blindly rank two submissions for the same task, where each submission was created by a different model [7]. The submissions are anonymized, for example as `Submission A` and `Submission B`, to reduce model-name and position bias.

So GDPval-AA is not:

```text
model output -> human expert occupational grader -> win rate vs human
```

It is closer to:

```text
model A agent run -> deliverable A
model B agent run -> deliverable B
Gemini judge compares A vs B
many pairwise comparisons -> Bradley-Terry fit -> Elo rating
```

The final Elo is computed from pairwise win/loss comparisons using a **Bradley-Terry model**, fitted by maximum likelihood estimation. Artificial Analysis says ties are excluded from the final Elo fit, and the rating scale is anchored to **GPT-5.1 Non-Reasoning = 1000 Elo**. They compute 95% confidence intervals by bootstrap resampling the match data and refitting the model many times.

This is slightly different from a simple chess-style online Elo update. In chess, ratings are often updated sequentially after each game. Here, AA collects a batch of pairwise judgments, fits a global Bradley-Terry model to estimate latent model strength, then reports that strength on an Elo-like scale. The practical interpretation is similar: a higher Elo model is expected to beat a lower Elo model more often in pairwise GDPval-AA judgments.

For example, if model X has a much higher GDPval-AA Elo than model Y, that does not mean X solved more tasks in an exact-match sense. It means that, across the judged pairwise comparisons, the judge tended to prefer X's submitted deliverables over Y's deliverables. The score is about **relative deliverable preference**, not deterministic correctness.

## 4.3.1 What does an Elo number mean?

Elo is a rating system originally used for chess. It turns many head-to-head wins and losses into one **relative strength number**.

The important point is that Elo is not an absolute percentage. A model with `1300 Elo` is not "65% correct" just because `1300 / 2000 = 65%`. Elo only becomes meaningful when compared with another Elo.

In a normal Elo system:

```text
same Elo      -> expected win rate is 50%
+100 Elo gap  -> higher-rated side wins about 64%
+200 Elo gap  -> higher-rated side wins about 76%
+300 Elo gap  -> higher-rated side wins about 85%
+400 Elo gap  -> higher-rated side wins about 91%
```

The standard expected-win formula is:

```text
P(A beats B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
```

So if:

```text
Model A = 1300 Elo
Model B = 1000 Elo
```

then:

```text
P(A beats B) = 1 / (1 + 10^((1000 - 1300) / 400))
             = 1 / (1 + 10^(-0.75))
             ~= 0.85
```

This means Model A is expected to beat Model B in about **85%** of pairwise comparisons under that evaluation setup.

In GDPval-AA, the comparison is not a chess game. It is:

```text
Model A deliverable vs. Model B deliverable
```

The "winner" is the submission preferred by the LLM judge. Therefore, a `1300 Elo` GDPval-AA model means:

```text
This model is estimated to produce deliverables that beat a 1000-Elo model
about 85% of the time in AA's GDPval-AA judging setup.
```

The number `1300` exists because AA chooses an anchor for the scale. Their methodology anchors **GPT-5.1 Non-Reasoning at 1000 Elo** [7]. Therefore:

```text
1300 Elo = 300 Elo stronger than the anchor model
```

It does not mean the model gets 1300 points out of a fixed maximum. It means the model is placed 300 rating points above the anchor based on pairwise comparison results.

Here are several examples:

| Model A Elo | Model B Elo | Gap | Expected A win rate | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 1300 | 1300 | 0 | 50% | Roughly equal under the judge. |
| 1300 | 1200 | +100 | 64% | A is modestly stronger. |
| 1300 | 1100 | +200 | 76% | A is clearly stronger. |
| 1300 | 1000 | +300 | 85% | A strongly beats the anchor. |
| 1300 | 1500 | -200 | 24% | A is much weaker than B. |

So when reading GDPval-AA, I always translate Elo into a sentence:

```text
Given this Elo gap, how often should the higher-rated model's deliverable
be preferred over the lower-rated model's deliverable?
```

That is the most useful mental model.

For the Artificial Analysis Intelligence Index, GDPval-AA is normalized before being mixed into the composite score. Their methodology describes this as:

```text
normalized GDPval-AA contribution = clamp((Elo - 500) / 2000)
```

So an Elo of `1400` contributes roughly `45%` to that component: `(1400 - 500) / 2000 = 0.45`. AA also freezes scores at the time a model is added to the Intelligence Index so that the composite score does not constantly move when the GDPval-AA pool changes.

This normalized number is only for the **Artificial Analysis Intelligence Index**. It is not the same thing as expected win rate.

For example:

```text
GDPval-AA Elo = 1300
normalized index contribution = (1300 - 500) / 2000 = 0.40 = 40%
```

But expected win rate against the 1000-Elo anchor is:

```text
Elo gap = 1300 - 1000 = 300
expected win rate ~= 85%
```

These two numbers answer different questions:

* **Normalized Elo contribution:** how much this benchmark contributes to AA's composite Intelligence Index.
* **Expected win rate:** how often this model should beat another model with a given Elo rating.

In short:

```text
Normalized Elo is for combining benchmarks.
Expected win rate is for comparing two models.
```

This design has strengths and weaknesses:

* **Strength:** pairwise comparison is easier than absolute scoring for open-ended deliverables.
* **Strength:** Elo/Bradley-Terry gives a relative ranking across many models without requiring every model to be compared against every other model on every task.
* **Weakness:** the result depends on the judge model's taste, blind-spot, and file-inspection ability.
* **Weakness:** if the judge is itself one of the frontier models, there can be style or self-preference bias, even with anonymization.
* **Weakness:** Elo is relative to the evaluated pool and anchor. It is not an absolute measure of "percent of work automated."

For our own model evaluation, I would treat GDPval-AA Elo as a useful external relative ranking, but I would not treat it as a replacement for domain-expert review when the output will be used in a real workflow.

## 4.4 How final metrics are computed

For the OpenAI-style GDPval metric, the scoring unit is a **task-level pairwise comparison** between a model deliverable and a human gold deliverable. The judge returns one of three outcomes:

```text
model better than human -> 1.0
model tied with human   -> 0.5
model worse than human  -> 0.0
```

From these outcomes, several metrics can be reported:

* **Win rate:** `count(score == 1.0) / number_of_graded_tasks`. This is the strictest interpretation.
* **Win + tie rate:** `count(score >= 0.5) / number_of_graded_tasks`. This answers whether the model reached or exceeded human deliverable quality.
* **Average pairwise score:** `mean(score)`. This gives partial credit for ties and is useful for automated grading reports.
* **Completion rate:** `completed_tasks / total_tasks`. This is not a quality score, but it is essential for debugging because GDPval agents can fail to submit files.
* **Breakdowns:** the same metrics grouped by sector, occupation, deliverable type, task duration, or file modality.

If each task is graded multiple times, the usual aggregation is:

1. Convert each judge result to `0`, `0.5`, or `1`.
2. Average repeated judgments for the same task/sample.
3. Average across all graded tasks.
4. Compute uncertainty, often with bootstrap confidence intervals over tasks or grader samples.

Inspect Evals reports an example automated-grader result as an average score with a 95% confidence interval and the number of tasks graded [6]. That is a grader-estimated metric, not the same as OpenAI's official human-expert leaderboard number. In their replication note, Inspect also points out that an automated-grader average can differ from official wins-only and wins-plus-ties numbers.

For GDPval-AA, the final metric is different:

```text
pairwise model-vs-model judgments -> Elo updates -> GDPval-AA Elo
```

This is why a model can have a strong GDPval-AA Elo while not having a directly comparable OpenAI-style win rate against human deliverables. Elo is relative to the pool of evaluated models and the judging/harness setup.

## 4.5 Why final metrics can disagree

When evaluating a new model, do not assume all GDPval numbers are directly comparable. Common sources of mismatch include:

* **Different harnesses:** OpenAI sampling, Inspect Evals, Artificial Analysis Stirrup, OpenHands, and custom internal agents may provide different tools.
* **Different tool access:** web search, Python, LibreOffice, browser, shell, code interpreter, file rendering, and package availability materially affect results.
* **Different sample counts:** one completion per task vs. three completions per task vs. best-of-N changes the final quality.
* **Different judging:** human expert preference, OpenAI automated grader, local rubric grader, or AA pairwise judge are not identical.
* **Different metric definitions:** wins only, wins plus ties, average ordinal score, task completion rate, and Elo all answer different questions.
* **Ungradable tasks:** OpenAI marks some public tasks as difficult or impossible for the automated grader because of internet needs, non-Python execution, fonts, or speech/audio limitations.

For professional reporting, always specify the dataset version, task count, harness, tool permissions, sampling policy, judge, and metric definition.

## 4.6 Which metrics actually matter?

GDPval runs often report many numbers. Some are true quality metrics, some are run-health metrics, and some are only dataset metadata. Mixing them up is one of the easiest ways to misread a GDPval result.

I usually divide the metrics into four levels:

```text
Level 1: Quality metrics     -> Did the model produce better work?
Level 2: Coverage metrics    -> How much of the benchmark was actually judged?
Level 3: Agent/process metrics -> How did the model behave while solving?
Level 4: Dataset/run metadata -> What was prepared or configured?
```

### Level 1: quality metrics

These are the most important metrics for model comparison.

**GDPval-AA Elo** is the key leaderboard metric when using Artificial Analysis style evaluation. It is a relative score derived from pairwise model-vs-model judgments. It answers:

```text
How often should this model's deliverable beat another model's deliverable?
```

Use this when comparing models under the same AA/Stirrup-style harness. Do not interpret it as a percentage correct.

**Win rate vs. reference/human deliverable** is the key OpenAI-style metric. It answers:

```text
How often does the model produce a deliverable preferred over the reference/human deliverable?
```

This is the cleanest quality metric if you are using pairwise comparison against human gold deliverables or a reference model deliverable.

**Win + tie rate** is a more forgiving version. It answers:

```text
How often does the model reach or exceed the reference quality bar?
```

This can be useful for deployment thinking because a tie may still mean the output is usable. However, it is less strict than wins only.

**Average pairwise score** treats outcomes as `win = 1`, `tie = 0.5`, `loss = 0`. It is useful for automated graders and confidence intervals, but it is less intuitive than win rate. If a report gives only an average score, I would ask whether ties were common and how many tasks were graded.

Professional viewpoint:

```text
For ranking models: prefer Elo or strict win rate.
For usability: also inspect win + tie rate.
For debugging grader variance: inspect average score and confidence interval.
```

### Level 2: coverage metrics

Coverage metrics do not tell us how good the model is, but they tell us whether the quality metric is trustworthy.

**Tasks attempted** means how many tasks were sent to the agent. For the public GDPval gold subset, the expected unique task count is usually `220`.

**Repeats / samples per task** means how many times each task was run. In one anonymized internal run, `num_repeats = 2`, so the prepared dataset has `440` examples. This is not 440 unique tasks; it is 220 tasks sampled twice.

**Tasks completed** means how many tasks produced a final submission. This is important because GDPval agents can fail before submission.

**Tasks graded** means how many submissions actually received a judge result. This can be lower than completed tasks if the judge fails, files are corrupted, deliverables are missing, or a task is ungradable.

**Ungradable / errored tasks** should always be reported. A model with high score on 120 graded tasks is not directly comparable to a model with slightly lower score on 219 graded tasks.

Professional viewpoint:

```text
Never trust a GDPval quality score without tasks attempted, tasks completed,
tasks graded, and error count.
```

### Level 3: agent and process metrics

These metrics explain *why* a model got a score. They are usually secondary for leaderboards, but very important for engineering.

**Average turns per task** tells us how many internal agent steps were used. Too few turns may mean the agent rushed or failed to inspect files. Too many turns may mean the model got stuck, looped, or struggled with tools.

**Tool-call counts** show whether the agent actually used shell, Python, browser, document conversion, image inspection, or file-writing tools. For GDPval, tool use is often not optional; a model that never opens reference files is probably producing shallow answers.

**Token usage** matters for cost and latency. In GDPval, higher token usage may indicate more careful work, but it can also indicate inefficient loops. It is not a quality metric by itself.

**Runtime / wall-clock time** matters for serving feasibility. It tells us whether a model can complete GDPval tasks within practical timeouts. It does not directly indicate correctness.

**Submission artifact count and file types** are critical for debugging. If the task asks for one PDF and one DOCX, submitting only text or the wrong file type should be treated as a serious failure even if the model's written explanation sounds good.

Professional viewpoint:

```text
Use process metrics to improve the harness and diagnose failures.
Do not use them alone to claim model quality.
```

### Level 4: dataset and run metadata

These metrics are easy to over-interpret.

In one prepared benchmark metadata file, the run reports values such as:

```text
Number of examples: 440
task_id unique_count: 220
sector unique_count: 9
occupation unique_count: 44
reference_files total_count: 522
rubric_json unique_count: 220
```

These are preparation metrics. They tell us the benchmark was expanded into the expected number of examples and that the dataset has the expected task/sector/occupation coverage.

They do **not** tell us whether the model performed well. For example:

```text
Number of examples = 440
```

only means:

```text
220 GDPval tasks x 2 repeats = 440 prepared samples
```

It does not mean 440 tasks were completed, 440 tasks were judged, or the model scored well.

Similarly:

```text
Number of tools: Average 0.0
Number of turns: Average 0.0
```

inside this prepared benchmark metadata should not be interpreted as "the agent used zero tools" or "the run was single-turn." In this data file, the examples only contain the prepared task payloads; the actual multi-turn agent behavior would be in rollout/request-response artifacts such as `evaluator_rollouts.jsonl`, which were not present in the copied folder.

Professional viewpoint:

```text
Dataset metadata is useful for validating preparation, not for evaluating model quality.
```

### My metric priority order

If I receive a GDPval report for a new model, I would read metrics in this order:

1. **Task coverage:** attempted, completed, graded, errored.
2. **Primary quality:** Elo, win rate, win + tie rate, or average pairwise score depending on the evaluation protocol.
3. **Uncertainty:** confidence interval or bootstrap range.
4. **Breakdowns:** sector, occupation, deliverable type, task duration, file modality.
5. **Failure modes:** missing files, wrong formats, corrupted artifacts, judge failures, timeouts.
6. **Efficiency:** average turns, runtime, tokens, cost.
7. **Preparation metadata:** examples, repeats, task IDs, sectors, occupations, reference-file counts.

For model selection, the most important line is usually:

```text
quality metric + confidence interval + number of graded tasks
```

For example:

```text
GDPval-AA Elo: 1320 ± 45, graded on 220 tasks
```

or:

```text
Average pairwise score: 0.47 ± 0.06, tasks graded: 219 / 220
```

For engineering/debugging, the most important line is usually:

```text
completion rate + artifact errors + average turns + timeout/judge failures
```

because a low GDPval score is often caused by missing deliverables, failed file conversion, or agent loops, not only weak reasoning.


# 5. Key findings from the GDPval paper

## 5.1 Frontier models are approaching expert deliverable quality

The most striking result is that frontier models are starting to approach professional human baselines on the GDPval gold subset. The paper reports that model performance has improved roughly linearly over time, and that the best current models can produce deliverables that experts sometimes judge as comparable to human expert work.

This does not mean "AI can replace the occupation." It means that for a bounded, well-specified digital task, a frontier model can sometimes produce an output that a professional grader finds competitive.

That distinction is important. GDPval measures task-level capability, not whole-job replacement.

## 5.2 Different models fail in different ways

The paper's error analysis is useful. Some models lose because they do not follow instructions carefully. Others produce attractive documents but make factual or calculation mistakes. Some fail to use reference data correctly. Others generate the right content but package it in a poor format.

This matches what we see in real agent workflows:

* **Accuracy is not enough** if the artifact is unusable.
* **Formatting is not enough** if the content is wrong.
* **Tool access is not enough** if the model does not verify its work.
* **Long reasoning is not enough** if the final deliverable ignores the user's constraints.

GDPval rewards the full loop: understand the task, use the context, create the artifact, inspect the result, and revise.

## 5.3 Reasoning effort and scaffolding help

The paper also shows that increasing reasoning effort and improving scaffolding can improve performance. In particular, prompting models to inspect generated files, check layouts, avoid formatting artifacts, and verify deliverables produced measurable gains.

This is an important lesson for agent builders. GDPval is not only a model benchmark; it is also a workflow benchmark. A better harness, better file inspection, best-of-N sampling, or a stronger review step can change the final quality.


# 6. Limitations

## 6.1 GDPval is broad, but not complete

GDPval covers 44 occupations and 9 sectors, but it is still only an initial slice of economic work. It focuses on digital knowledge tasks that can be packaged into prompts, reference files, and deliverables.

It does not cover manual labor, physical-world tasks, deeply interactive workplace collaboration, proprietary internal systems, or work that depends heavily on tacit organizational context.

## 6.2 The tasks are mostly one-shot

Real work is often interactive. A professional asks clarifying questions, negotiates requirements, checks assumptions, and gets feedback from stakeholders. GDPval tasks provide the context up front, so they are more self-contained than many real workplace tasks.

This makes the benchmark easier to run consistently, but it also means GDPval may overestimate performance on ambiguous, under-specified work.

## 6.3 Grading is expensive and subjective

The benchmark's strongest feature is also its bottleneck: expert grading. High-quality professional comparison takes time and money. Automated grading helps, but it remains a proxy.

For open research, this creates a tradeoff. We can iterate quickly with automated rubrics, but the most meaningful results still require careful human evaluation.


# 7. How GDPval runs in practice

## 7.1 Endpoint-to-deliverable flow

If the user gives an inference endpoint or model server, GDPval does not simply send 220 text prompts and compute exact match. A practical runner needs an **agent loop** around the endpoint.

A typical flow is:

1. Load one GDPval task from the dataset.
2. Create a sandbox workspace for that task.
3. Download or mount the reference files.
4. Send the prompt, file inventory, and system instructions to the model.
5. Let the model call tools through the harness: shell, Python, file read/write, browser or web search if allowed, document conversion, image inspection, etc.
6. Collect the model's final deliverable text and deliverable files.
7. Save the task result into a submission dataset.
8. Upload the submission dataset to Hugging Face or pass it to a local grader.
9. Run pairwise or rubric-based judging against gold deliverables.
10. Aggregate the final scores.

This is why GDPval is sensitive to infrastructure details. A weak model with a strong file-inspection loop can outperform the same model in a text-only setup. A strong model can fail if the harness cannot create PDFs, inspect PowerPoint renders, or upload the expected deliverable path.

## 7.2 Running through Inspect Evals

The public Inspect Evals implementation is currently one of the clearest reproducible ways to run the 220-task gold subset [6]. Installation is:

```bash
pip install 'inspect-evals[gdpval]'
```

Or from the repository:

```bash
uv sync --extra gdpval
```

A minimal run looks like:

```bash
uv run inspect eval inspect_evals/gdpval --model openai/gpt-5-nano
```

For an OpenAI-compatible or custom endpoint, the exact model string depends on how Inspect AI is configured. In practice, you usually need to provide:

* The model provider/model name used by Inspect.
* The endpoint base URL if it is an OpenAI-compatible server.
* The API key or dummy API key expected by the client.
* Any generation parameters such as temperature, max output tokens, reasoning effort, or timeout.

The runner creates an output folder containing deliverable files and metadata. Inspect Evals can also upload the results to Hugging Face:

```bash
hf auth login
uv run inspect eval inspect_evals/gdpval -T upload_to_hf=True
```

The generated Hugging Face dataset is then used for grading. Earlier instructions required submitting the uploaded dataset through the OpenAI grading form. The OpenAI Evals page now says rubrics and gold deliverables are open-sourced, so teams can also run their own grading workflows [3].

## 7.3 What the submission dataset must contain

A GDPval submission is basically a table of task outputs plus the files generated by the model. A robust submission should preserve:

* `task_id`: unique task identifier.
* `sector` and `occupation`: useful for breakdowns.
* `prompt`: the original request.
* `reference_files`: file names or URIs used by the task.
* `deliverable_text`: final text response, if any.
* `deliverable_files`: paths or URIs to generated PDFs, spreadsheets, slides, images, archives, or other artifacts.
* Runtime metadata: model name, endpoint, sampling parameters, tool permissions, timestamps, token usage, cost, and any failures.

The most common evaluation bug is not model reasoning. It is broken artifact plumbing: files are created in the wrong directory, extra temporary files are uploaded, the final answer references a file that was not included, or the submitted PDF differs from the file the model inspected.

## 7.4 How to run multi-turns

There are two different meanings of "multi-turn" here.

First, **official GDPval tasks are one-shot in the user-facing sense**. The task prompt provides the request and context up front. The model is not supposed to ask the professional user clarifying questions over multiple external turns. OpenAI explicitly lists this as a limitation: real work is more interactive than the current benchmark.

Second, **GDPval agents can be multi-turn internally**. In an agent harness, the model may take many turns with tools:

```text
model thinks -> calls shell
model observes files -> writes script
model opens spreadsheet -> creates chart
model renders PDF -> inspects PNG pages
model fixes formatting -> submits final deliverable
```

This is the right way to run GDPval for an endpoint. The benchmark prompt is one task, but the solver loop can take many model-tool turns until it submits. For endpoint evaluation, you need to decide and log:

* Maximum turns per task.
* Maximum wall-clock time per task.
* Maximum tokens or budget per task.
* Which tools are available.
* Whether web access is allowed.
* Whether the model can see rendered images of generated documents.
* What command or API marks final submission.

Artificial Analysis' GDPval-AA explicitly uses an agentic loop through Stirrup with shell and web access [5]. Inspect Evals uses a Docker sandbox and gives the model bash/Python-style tooling because GDPval deliverables often require file generation and inspection [6].

## 7.5 An anonymized internal run structure

I also inspected one internal GDPval agentic run bundle. To avoid exposing local paths, model identifiers, endpoints, credentials, or log-derived deployment details, I describe the structure using sanitized names only.

The bundle contains the prepared GDPval benchmark file, benchmark metadata, runtime metadata, run configuration, and logs. The most useful artifacts for understanding the run are:

```text
<run>/preprocessed_datasets/benchmark.jsonl
<run>/preprocessed_datasets/benchmark_metrics.json
<run>/artifacts/run_config.yml
<run>/artifacts/run_times/runtime_*.json
<run>/logs/*.log
```

From the prepared benchmark metadata, this run prepared:

```text
benchmark: gdpval
unique tasks: 220
num_repeats: 2
total examples: 440
unique sectors: 9
unique occupations: 44
reference file entries: 522
```

From the sanitized run configuration, the evaluation had the following shape:

```text
policy endpoint: internal OpenAI-compatible chat endpoint
policy model: evaluated target model
judge model: configured pairwise judge model
agent harness: Stirrup-style GDPval agent
agent concurrency: configured parallel workers
agent_max_turns: 100
reward_mode: comparison
num_repeats: 2
```

This means each GDPval task is not just a single prompt completion. The policy model is wrapped by the Stirrup-style agent. The agent can take multiple internal turns, use tools, create files, inspect results, and eventually submit deliverables. The judge then compares the generated deliverables against reference deliverables.

One important caveat: this bundle does **not** include `evaluator_rollouts.jsonl` or a saved request/response cache, so I cannot recover the exact model messages and tool calls for each task. The examples below are therefore reconstructed from the prepared task prompts/rubrics and the sanitized run configuration. They are still useful because they show the multi-turn structure that the harness expects, but they should not be read as verbatim trajectories from this run.

### Example A: Audit sampling workbook

Task:

```text
task_id: 83d10b06-26d1-4636-a32c-23f92c57f30b
sector: Professional, Scientific, and Technical Services
occupation: Accountants and Auditors
reference file: Population v2.xlsx
deliverable: Excel workbook named Sample
```

The user asks the agent to review Anti-Financial Crime Risk Metrics for Q2 and Q3 2024, calculate a required audit sample size at 90% confidence and 10% tolerable error, perform quarter-on-quarter variance analysis, select samples according to stated criteria, and create a workbook with a `Sample Size Calculation` tab.

A realistic multi-turn progression is:

1. **Read task and inspect files.** The agent lists the workspace, finds `Population v2.xlsx`, and opens workbook metadata.
2. **Load spreadsheet data.** It uses Python, `openpyxl`, or `pandas` to inspect sheets, headers, data row count, and Q2/Q3 metric columns.
3. **Compute audit sample size.** It calculates population size `N`, uses the attribute sampling formula with `z = 1.645`, `p = 0.5`, `e = 0.10`, and finite population correction.
4. **Perform variance analysis.** It computes Q3 minus Q2 or the requested quarter-on-quarter variance and writes the result into the expected column.
5. **Select sample rows.** It filters rows so each selected sample satisfies at least one stated criterion and the whole selected set covers all criteria.
6. **Create workbook.** It writes the selected sample worksheet and a `Sample Size Calculation` worksheet.
7. **Validate rubric constraints.** It reopens the workbook to check sheet names, column order, row values, formulas/workings, and filename.
8. **Submit deliverable.** It returns the final Excel workbook as the task output.

The important point is that the benchmark is testing both accounting logic and file craftsmanship. A model can reason correctly about sample size but still lose points if the workbook is named incorrectly, the worksheet title is wrong, or copied columns do not exactly match the source.

### Example B: Prepaid amortization workbook

Task:

```text
task_id: 7d7fc9a7-21a7-4b83-906f-416dea5ad04f
sector: Professional, Scientific, and Technical Services
occupation: Accountants and Auditors
reference files: COA.xlsx plus January-April prepaid expense PDFs and prepaid insurance PDF
deliverable: single .xlsx workbook
```

The user asks for a detailed amortization schedule for Aurisic's prepaid expenses and insurance through April 2025. The deliverable must contain three tabs:

```text
Prepaid Summary
Prepaid Expenses (Account #1250)
Prepaid Insurance (Account #1251)
```

A realistic multi-turn progression is:

1. **Inventory references.** The agent identifies one chart-of-accounts workbook and several invoice PDFs.
2. **Extract PDF data.** It converts or parses each PDF to recover vendor names, invoice dates, service periods, prepaid amounts, and insurance details.
3. **Map accounts.** It uses `COA.xlsx` to confirm account numbers and labels for prepaid expenses and prepaid insurance.
4. **Build amortization logic.** For each invoice, it calculates monthly expense, amortization period, year-to-date amortization, and remaining balance by month.
5. **Create detailed tabs.** It writes separate schedules for account `1250` and account `1251`, sorted by vendor or the task's requested structure.
6. **Create summary tab.** It links summary totals to detailed sheets with formulas rather than hard-coded values.
7. **Check exact expected values.** The rubric includes numeric expectations, such as an April 2025 GL balance for Prepaid Expenses of `$559,377.61`, so the agent should verify formulas against those targets.
8. **Submit workbook.** It saves a single `.xlsx` file and ensures all three sheets are present.

This example shows why GDPval is hard for endpoint evaluation. The model needs a long loop over PDFs, spreadsheets, accounting rules, formulas, and final-file validation. If the endpoint has weak tool use, weak PDF extraction, or short context, this task can fail even if the base model is strong.

### Example C: Ergonomics checklist and action tracker

Task:

```text
task_id: 27e8912c-8bd5-44ba-ad87-64066ea05264
sector: Government
occupation: Administrative Services Managers
reference files: none, but prompt points to an NIH ergonomics checklist URL
deliverables: one PDF checklist and one Word .docx action-items document
```

The user asks for two deliverables:

```text
1. Workstation Ergonomics Checklist - PDF, no more than five pages
2. Organizational Action Items - Word document with an action-item tracking table
```

The checklist must focus only on office chair, keyboard/mouse, and work surface setup. It should use a credible source, such as the NIH workstation ergonomics self-assessment, and include fields like employee name, position, and assessment metadata.

A realistic multi-turn progression is:

1. **Understand deliverable split.** The agent identifies that the final answer requires two separate files, not one combined report.
2. **Fetch or summarize source.** If web is enabled, it retrieves the NIH checklist; if not, it uses known ergonomic principles and cites the prompt-provided source.
3. **Scope the checklist.** It removes unrelated topics like breaks, laptops, accessories, hot-desking, or general wellness, because the rubric explicitly limits the scope.
4. **Draft PDF content.** It creates a concise checklist with goal/purpose, employee fields, and assessment items for chair, keyboard/mouse, and work surface.
5. **Draft DOCX tracker.** It creates a Word table for organizational action items, including owner, issue, priority, action, due date, status, and follow-up notes.
6. **Render and inspect.** It converts the checklist to PDF, checks page count is no more than five, and verifies the Word file opens.
7. **Submit both files.** It submits exactly one PDF and exactly one `.docx` file.

This example is less numerically complex than the accounting tasks, but it is format-sensitive. The agent can lose if it submits only a PDF, includes too many pages, covers forbidden topics, or fails to create a real `.docx` file.

### What these examples reveal about multi-turn GDPval

Across these examples, the internal loop has the same shape:

```text
read prompt
inspect reference files or web source
extract structured data
reason over task-specific requirements
create deliverable files
re-open or render outputs
validate against rubric-like constraints
submit final artifacts
```

For this anonymized agentic setup, the key operational settings are `agent_max_turns=100`, parallel task execution, and `num_repeats=2`. That means the benchmark gives each task enough room for a long agent trajectory, runs many tasks in parallel, and samples each task twice. In practice, the quality of the run depends heavily on whether the saved artifacts preserve the full trajectory, whether the sandbox can handle Office/PDF conversion, and whether the agent actually verifies its deliverables before submission.

For future runs, I would strongly recommend preserving the following outputs in the downloaded artifact folder:

```text
evaluator_rollouts.jsonl
evaluator_rollouts_materialized_inputs.jsonl
evaluator_rollouts_aggregate_metrics.json
request/response logs for at least a small sample
submitted deliverable files
raw judge responses
```

Without those files, we can describe the intended multi-turn structure, but we cannot audit exact per-turn model behavior.

## 7.6 Practical endpoint issues

When evaluating a new model endpoint, I would check the following before trusting the final metric:

* **Context length:** GDPval prompts plus reference-file summaries can be long. Truncation silently hurts performance.
* **File ingestion:** If the model cannot read PDFs, spreadsheets, images, or slides, the harness must convert them into accessible text/images.
* **File generation:** Many tasks need `.pdf`, `.pptx`, `.docx`, `.xlsx`, `.png`, `.zip`, or similar outputs. The sandbox must include the right libraries.
* **Visual inspection:** For slides, PDFs, and spreadsheets, render to PNG and let the model inspect pages before submitting.
* **Timeouts:** Some tasks are long. The Inspect docs mention sandbox build time and long grader calls; local endpoints also need generous request timeouts.
* **Tool-call compatibility:** Some endpoints do not support function calling, parallel tool calls, image inputs, or long assistant messages in the same way as frontier APIs.
* **Determinism:** GDPval scores have variance. Run repeated samples if the budget allows.
* **Submission contract:** The judge only sees submitted files/text. Anything left in scratch space or linked by a broken path does not count.


# 8. How to explore GDPval

## 8.1 Dataset

The public gold subset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/openai/gdpval
```

It contains 220 real-world knowledge-work tasks across 44 occupations. Each task includes a text prompt and supporting reference files.

Using the Hugging Face `datasets` library, the basic exploration flow should look like:

```python
from datasets import load_dataset

dataset = load_dataset("openai/gdpval")
print(dataset)
print(dataset["train"][0].keys())
```

From there, the useful things to inspect are:

* What occupation and sector the task belongs to.
* How many reference files are attached.
* What kind of deliverable is expected.
* Whether the task is mostly reasoning, formatting, multimodal understanding, or file manipulation.

## 8.2 Evaluation mindset

When using GDPval, I would avoid treating it like a normal QA benchmark. A better evaluation loop is:

1. Let the model or agent generate the deliverable.
2. Render or open the artifact when possible.
3. Run any deterministic checks you can add locally.
4. Use the provided rubrics or automated grader for a rough signal.
5. Reserve strong claims for human expert comparison.

This mirrors how professional work is actually reviewed: not just "did the model answer," but "would I trust this output enough to use it?"


# 9. My takeaways

GDPval is valuable because it pushes evaluation toward the messy middle between academic benchmarks and real deployment. It does not solve every evaluation problem, but it asks the right kind of question: can models produce economically useful work products under realistic constraints?

For LLM infrastructure and agent research, I think GDPval is especially useful in three ways:

* It highlights the importance of **artifact quality**, not just reasoning traces.
* It exposes the need for **tooling and verification loops**, especially for files, slides, spreadsheets, and visual outputs.
* It provides a better lens for measuring **human-AI collaboration**, where the model may save time even if a human still reviews or edits the final result.

The benchmark also reminds us to be careful. A high GDPval score does not mean an occupation is automated. It means a model is getting better at a sampled set of well-scoped digital tasks. That is still a big deal, but the real-world impact depends on workflow integration, oversight, trust, cost, latency, and the many informal parts of work that benchmarks still struggle to capture.


# References

[1] OpenAI, **GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks**, arXiv:2510.04374.  
<https://arxiv.org/html/2510.04374v1>

[2] OpenAI, **openai/gdpval Dataset**, Hugging Face.  
<https://huggingface.co/datasets/openai/gdpval>

[3] OpenAI, **GDPval Grading**, OpenAI Evals.  
<https://evals.openai.com/gdpval/grading>

[4] OpenAI, **Measuring the performance of our models on real-world tasks**.  
<https://openai.com/index/gdpval>

[5] Artificial Analysis, **GDPval-AA Leaderboard**.  
<https://artificialanalysis.ai/evaluations/gdpval-aa>

[6] UK Government BEIS Inspect Evals, **GDPval**.  
<https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/gdpval/>

[7] Artificial Analysis, **Intelligence Benchmarking Methodology**.  
<https://artificialanalysis.ai/methodology/intelligence-benchmarking>
