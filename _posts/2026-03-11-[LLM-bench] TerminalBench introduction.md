---
title: TerminalBench introduction
date: 2026-03-11
categories: [LLM, bench]
tags: [LLM, agents]
description: An introduction to TerminalBench, a benchmark designed to evaluate how well AI agents complete complex, real-world tasks in terminal environments.
---

# 1. Why do we need TerminalBench?

## 1.1 The rise of terminal-based AI agents

When we think about how software engineers and system administrators interact with computers, the terminal is often the tool of choice. It provides fine-grained control, is highly versatile, and is easily sandboxed. Naturally, as AI agents like Cursor, Devin, and OpenHands become more advanced, they are increasingly relying on the terminal to perform actions.

The terminal is a text-based interface, which perfectly matches the primary modality of Large Language Models (LLMs). This makes it an ideal environment for AI agents to execute commands, write code, configure systems, and perform data analysis.

## 1.2 The limitations of current agents

Despite their potential, current AI agents still struggle to fully exploit the power of the terminal. They often fail when required to chain together multiple complex actions, reason over long contexts, or act independently within strict constraints. Furthermore, safely executing sensitive tasks without human supervision remains a significant challenge.

While existing benchmarks evaluate LLMs on coding or general reasoning, they often fall short in measuring an agent's ability to navigate the messy, interactive, and stateful world of a real command-line interface (CLI).

## 1.3 Bridging the evaluation gap

To build better agents, we need better ways to measure their capabilities in realistic environments. We need an evaluation framework that tests whether an agent can handle the nuances of a real terminal—from installing dependencies and resolving conflicts to interacting with complex CLI tools. This is where **TerminalBench** comes in.


# 2. What is TerminalBench?

## 2.1 Overview and design philosophy

TerminalBench is a comprehensive evaluation framework and benchmark designed to quantify how well AI agents perform complex tasks in the terminal. Developed by researchers from the Harbor Framework and various institutions, it provides a curated dataset of challenging, human-verified tasks.

The core philosophy of TerminalBench is realism and difficulty. Each task is inspired by real-world workflows and comes with its own dedicated Docker environment, a human-written reference solution, and a comprehensive set of test cases for automated verification.

## 2.2 Task diversity and a deep dive into examples

TerminalBench covers a wide range of domains, including software engineering, machine learning, cybersecurity, and data science. To truly understand how TerminalBench works, we need to look at the anatomy of a task. 

Every task in the dataset is defined by a few core components:
*   **The Environment (`Dockerfile`):** The isolated sandbox where the task takes place.
*   **The Prompt (`task.yaml`):** The natural language instruction given to the agent.
*   **The Agent's Actions:** The actual bash commands or keystrokes the agent generates inside the `tmux` session.
*   **The Verification (`run-tests.sh`):** The deterministic script the harness runs afterward to grade the task.

Let's break down three specific examples from the benchmark to see how this plays out in practice:

### Example 1: Using new CLI tools (The "New Encrypt Command" Task)
*   **The Environment:** A lightweight Docker container pre-loaded with a custom, undocumented binary executable (`my_encryptor`) and a plaintext file (`secret.txt`).
*   **The Prompt:** *"Use the provided `my_encryptor` binary to encrypt `secret.txt` with the password 'tbench2026'. Save the output as `encrypted.bin`."*
*   **The Agent's Actions:** The agent first runs `ls` to see the files. Realizing it doesn't know the tool's syntax, it executes `./my_encryptor --help` to read the manual from standard output. It parses the required flags from the output and executes `./my_encryptor -in secret.txt -out encrypted.bin -pass tbench2026`.
*   **The Verification:** Once the agent signals completion, the harness runs `run-tests.sh` inside the container. This script uses a known decryption key to reverse `encrypted.bin` and asserts that the byte-for-byte content matches the original `secret.txt`. If it matches, the task passes.

### Example 2: Compiling complex software (Build Linux Kernel with QEMU)
*   **The Environment:** A barebones Ubuntu image. The agent has root access but must figure out which system packages are missing to complete a complex build process.
*   **The Prompt:** *"Download the Linux kernel source, configure it, compile it from source, and verify it boots using QEMU."*
*   **The Agent's Actions:** The agent typically starts strong, running `apt-get update && apt-get install build-essential qemu-system-x86`. It downloads the tarball and extracts it. However, to edit a configuration file, the agent might decide to run `nano config`. Because `nano` is a full-screen interactive TUI (Text User Interface), the terminal floods with ANSI escape sequences. The LLM struggles to parse this visual interface, gets stuck in the editor, and ultimately gives up or times out.
*   **The Verification:** The test script checks for the existence of the compiled `bzImage`. It then runs a headless QEMU command to boot the image, capturing the serial console output to confirm the kernel successfully reaches a specific runlevel. In the case of the `nano` failure, the image is never built, resulting in a fail.

### Example 3: Scientific computing (Raman Fitting)
*   **The Environment:** A Python-ready Docker container containing a CSV file (`spectrum.csv`) of a Raman spectrum (light intensity data).
*   **The Prompt:** *"Write a Python script to fit the two main peaks in `spectrum.csv` and output the peak centers and widths to `fit_results.json`."*
*   **The Agent's Actions:** The agent successfully writes a script using `scipy.optimize.curve_fit`, runs it, and generates the JSON file. However, the data provided in the CSV is in wavelengths (nm) rather than the standard wavenumbers (cm⁻¹). The agent blindly fits the data without checking the physical units, resulting in physically nonsensical parameters (like negative amplitudes or massive peak widths). It asserts the task is done.
*   **The Verification:** A Python test script (`test_outputs.py`) loads `fit_results.json` and checks if the fitted parameters fall within a scientifically valid, expected range. Since the agent didn't convert the units or notice the impossible physics, the parameters fail the bounds check, and the task is marked as a failure. This highlights how TerminalBench tests deeper domain reasoning, not just code generation.

## 4.4 A Real-World Deep Dive: The `aimo-airline-departures` Task

To make this completely concrete, let's look at a real evaluation trace from a TerminalBench run. In this specific trial, the `terminus-2` agent (powered by a custom model, `nemotron-super-rl`) was tasked with solving a complex logic puzzle using Python.

### The Setup and Prompt
The agent is dropped into a clean `/app` directory inside a Debian-based Docker container. It is given the following prompt:

> **Task Description:**
> Three airline companies operate flights from Dodola island. Each company has a different schedule of departures:
> - The first company departs every 100 days
> - The second company departs every 120 days
> - The third company departs every 150 days
> 
> What is the greatest positive integer *d* for which it is true that there will be *d* consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?
> 
> Your task is to:
> 1. Create a Python script named `/app/airline_analysis.py` that solves the problem to find the maximum gap.
> 2. Write the final answer (just the number) to a file called `/app/results.txt`.

### The Agent's Inference Loop (Episodes)
The `terminus-2` agent reads this prompt and the empty terminal screen, and begins its reasoning process. In TerminalBench, the agent's interaction is broken down into **episodes** (or steps). In this specific trial, the agent took **12 episodes** to complete the task. 

Let's look at the raw trajectory data to see exactly how the agent handled this across a few key episodes:

**Episode 1 (Reasoning and Coding):** 
The LLM first reasons through the math in its internal scratchpad. It identifies that the Least Common Multiple (LCM) of 100, 120, and 150 is 600. It deduces that it needs to find the minimum possible maximum gap over all possible phase offsets for the three airlines within a 600-day period. Instead of using an interactive editor like `nano` (which can be tricky for LLMs), the agent cleverly uses a bash heredoc to write the script directly to the file system. 

Here is the exact JSON payload the agent generated to execute this:
```json
{
  "analysis": "We are in the /app directory with a clean terminal. Need to create a Python script to solve the airline departure problem and output the answer to results.txt...",
  "plan": "1. List directory contents to confirm we're in /app. 2. Create the Python script airline_analysis.py using cat or a text editor. 3. Run the script... 4. Verify the result...",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cat > /app/airline_analysis.py << 'EOF'\nimport itertools\ndef main():\n    periods = [100, 120, 150]\n    LCM = 600\n    counts = [LCM // p for p in periods]\n    min_max_gap = LCM\n    # ... (brute force loops omitted for brevity) ...\n    with open('/app/results.txt', 'w') as f:\n        f.write(str(min_max_gap))\nif __name__ == '__main__':\n    main()\nEOF\n",
      "duration": 0.5
    },
    {
      "keystrokes": "python3 /app/airline_analysis.py\n",
      "duration": 2.0
    },
    {
      "keystrokes": "cat /app/results.txt\n",
      "duration": 0.1
    }
  ]
}
```

**Episodes 3-11 (Waiting and Polling):** 
Because the brute-force Python script processes 1.8 million iterations, it takes several seconds to run. The agent observes that the terminal is hanging and hasn't printed the output of the `cat` command yet. 

Instead of panicking or hallucinating an answer, the agent enters a polling loop. Here is the agent's internal analysis during Episode 3:
> *"Analysis: The terminal output shows that after the python script execution, the cat command was issued but we haven't seen the output of either. Possibly the python script is still running... We should first check if results.txt exists and what it contains by running cat again... Plan: 1. Send a newline to get a fresh prompt."*

The agent then sends empty keystrokes (`\n`) with a 1.0-second wait duration, repeatedly checking the terminal state until the script finishes.

**Episode 12 (Verification and Submission):** 
The script finally finishes, and the terminal prints the output of `cat /app/results.txt` (which is `20`). The agent reads this, verifies that the file was created and contains a valid integer, and then submits the task as complete.

### The Verification Phase
Once the agent submits, the `harbor` harness takes over the frozen container and runs the hidden verification suite. 

If we look at the raw `test-stdout.txt` logs from the harness, we see exactly how rigorous this is. The harness doesn't just ask an LLM if the answer looks right. Instead, it:
1.  Installs necessary system dependencies (`curl`, `uv`).
2.  Creates an isolated Python virtual environment and installs `pytest`.
3.  Runs a hidden test script (`test_outputs.py`) against the agent's generated files.

The output of the test runner looks like this:
```text
============================= test session starts ==============================
platform linux -- Python 3.13.1, pytest-8.4.1, pluggy-1.6.0
rootdir: /tests
collected 4 items

../tests/test_outputs.py ....                                            [100%]

==================================== PASSES ====================================
PASSED ../tests/test_outputs.py::test_required_files[/app/airline_analysis.py-Python script]
PASSED ../tests/test_outputs.py::test_required_files[/app/results.txt-results file]
PASSED ../tests/test_outputs.py::test_script_runs
PASSED ../tests/test_outputs.py::test_answer
============================== 4 passed in 6.73s ===============================
```

The verifier checks four distinct things: Did the agent create the script? Did it create the results file? Does the script execute without throwing syntax errors? And finally, is the mathematical answer in `results.txt` correct? Because all four tests passed, the harness records a strict `1.0` reward for this trial. 

This real-world trace perfectly illustrates the power of TerminalBench: it tests mathematical reasoning, coding ability, terminal fluency, and self-verification, all evaluated deterministically without human or LLM bias.

## 2.3 The statistics of TerminalBench

TerminalBench is actively evolving. The current major versions include:
*   **Terminal-Bench 1.0 (Core-v0):** The initial release featuring 80 diverse tasks.
*   **Terminal-Bench 2.0:** An expanded and refined set of 89 high-quality, hard tasks.
*   **Domain-specific subsets:** Such as Terminal-Bench Science, focusing on scientific computing workflows.

The benchmark is designed to be difficult. According to recent evaluations, even frontier models and state-of-the-art agents score less than 65% on Terminal-Bench 2.0, indicating that there is still significant room for improvement in agentic capabilities.

## 2.4 The Execution Harness and Agents

TerminalBench isn't just a dataset; it's a complete execution harness called `harbor`. This harness manages the lifecycle of the evaluation:
1.  **Orchestration:** It spins up multi-container Docker environments to safely sandbox the agents.
2.  **Integration:** It connects the language model to the terminal environment. It supports various integration methods, including installing the agent directly into the container, direct Python API integration, or using the Model Context Protocol (MCP).
3.  **Evaluation:** It logs agent actions and runs the verification scripts to check the final container state against the expected outcome.

### The Role of the Agent and a Deep Dive into Terminus

You might wonder: if we are evaluating an LLM, why do we need an "agent" layer in the middle? Is the agent itself an LLM?

**To clarify: The agent (like `terminus-2`) is NOT an LLM. It is simply a Python wrapper—a piece of software—that connects the LLM you want to evaluate to the terminal environment.**

An LLM on its own just takes text in and spits text out. It doesn't know how to open a terminal, type keys, or read the screen. The **agent** is the software wrapper that gives the LLM hands and eyes. It is responsible for:
*   **Reading** the current state of the terminal (via `tmux`).
*   **Formatting** that state into a text prompt the LLM can understand.
*   **Sending** that prompt to the LLM (e.g., via an API call to GPT-4, Claude, or a local model).
*   **Parsing** the text response that comes back from the LLM.
*   **Translating** that text response into actual keystrokes or bash commands executed in the container.

To ensure a level playing field when comparing different *models*, the creators of TerminalBench introduced **Terminus** (and its iterations like `terminus-2`). 

Terminus is a research-preview agent specifically designed to evaluate how well language models can power autonomous agents in terminal environments. It is actually one of the highest-performing agents on the benchmark, but its design is intentionally minimalist. Here is what makes Terminus unique:

1.  **Mono-Tool Design:** Unlike complex coding agents that have a dozen different tools (a specific tool for reading files, another for writing, another for searching), Terminus uses a **single tool**: an interactive `tmux` session. It simply sends keystrokes and reads the screen. This forces the LLM to figure out *how* to do things using standard Linux utilities (like using `cat`, `nano`, or `grep`) rather than relying on custom agentic crutches.
2.  **Model-Agnostic:** While agents like Claude Code are locked to specific model providers, Terminus uses `LiteLLM` under the hood. This means it can be paired with virtually *any* model—from OpenAI's GPT-4o to locally hosted open-source models (as shown in the setup guide below).
3.  **Autonomy-First:** Terminus operates entirely autonomously. It never pauses to ask the user for input or clarification, and it doesn't impose artificial guardrails on what terminal commands it can run. This makes it perfect for sandboxed, automated evaluations.
4.  **Independent Execution:** Terminus runs in a separate Python process outside the task container, remotely connecting to the Docker environment. This is crucial because it means Terminus can still operate even if the task environment has broken networking, corrupted Python installations, or other chaotic states that would break agents installed directly inside the container.

By using a baseline agent like Terminus, researchers can isolate and evaluate the raw reasoning and coding capabilities of the underlying LLM itself.

### Alternative Agents

While Terminus is the default for benchmarking base models, the `harbor` framework is designed to be agent-agnostic. You can absolutely replace `terminus-2` with other, more complex agent frameworks to see how different agent architectures perform on the same tasks. 

Some alternative agents you might evaluate using TerminalBench include:
*   **OpenHands (formerly OpenDevin):** A powerful open-source agent designed for software engineering tasks.
*   **Goose:** An open-source developer agent that can be integrated via the Model Context Protocol (MCP).
*   **Claude Code:** Anthropic's CLI-based coding agent (though evaluating closed-source, proprietary agents often requires different integration strategies, like installing them directly into the task container).
*   **Custom Agents:** You can build your own Python-based agent and plug it directly into the `harbor` harness to test your own agentic architectures.


# 3. How to set up and run TerminalBench locally

While TerminalBench is available via standard package managers, running it against local or custom models often requires using the underlying `harbor` execution framework directly. Here is a step-by-step guide to setting it up using `uv` and running evaluations against a locally hosted model (e.g., via vLLM).

## 3.1 Installation and dependencies

First, we'll use `uv`, an extremely fast Python package installer, to manage the environment and install the `harbor` CLI from a specific branch that supports local execution.

**1. Install `uv`:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the harbor source code:**
```bash
git clone https://github.com/vegaluisjose/harbor.git
```

**3. Install harbor:**
```bash
cd harbor
uv tool install .
```

## 3.2 Running the hello-world task

To verify your setup, you can run a simple `hello-world` task. In this example, we are using the `terminus-2` agent and pointing it to a local OpenAI-compatible endpoint (running on port `8000`).

```bash
OPENAI_API_KEY=dummy harbor run \
   -d hello-world \
   --n-concurrent 4 \
   -m openai/your-custom-model-name \
   -a terminus-2 \
   --agent-kwarg api_base=http://localhost:8000/v1 \
   --agent-kwarg 'model_info={"max_input_tokens": 131072, "max_output_tokens": 131072, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}' \
   --agent-kwarg temperature=1.0 \
   --agent-kwarg top_p=0.95 \
   --agent-kwarg max_tokens=32000
```

## 3.3 Running the Artificial Analysis (AA) subset

Once you've confirmed the basic setup works, you can run a larger batch of tasks, such as the Artificial Analysis (AA) subset. You can increase the `--n-concurrent` flag to speed up the evaluation if your local inference server can handle the throughput.

```bash
OPENAI_API_KEY=dummy harbor run \
   --jobs-dir jobs \
   --path ./tasks/aa-subset \
   --n-concurrent 16 \
   -k 1 \
   -a terminus-2 \
   -m openai/your-custom-model-name \
   --agent-kwarg api_base=http://localhost:8000/v1 \
   --agent-kwarg 'model_info={"max_input_tokens": 131072, "max_output_tokens": 131072, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}' \
   --agent-kwarg temperature=1.0 \
   --agent-kwarg top_p=0.9 \
   --agent-kwarg max_tokens=32000
```

This configuration allows you to thoroughly test your models using the `terminus-2` agent without relying on external API providers.

## 3.4 Contributing

TerminalBench is a community-driven effort. If you have a complex terminal task that you think would stump current AI agents, the project welcomes contributions. They provide a quickstart guide for adding new tasks, complete with Dockerfiles, reference solutions, and test scripts.


# 4. Understanding Inference and Evaluation

To truly appreciate TerminalBench, it helps to understand exactly what happens under the hood when you run the `harbor run` command. How does the model actually interact with the terminal, and how do we know if it succeeded?

## 4.1 The Inference Loop (Agent-Environment Interaction)

TerminalBench doesn't just ask the LLM a question and parse a single answer. It facilitates a continuous, stateful interaction loop. Here is exactly how the agent wrapper connects the LLM to the terminal:

1.  **Environment Initialization:** For each task, the `harbor` harness spins up an isolated Docker container. This container holds the specific operating system, files, and pre-installed tools required for the task. Inside this container, a `tmux` (terminal multiplexer) session is started.
2.  **The Agent Bridge:** The agent (e.g., `terminus-2`) runs as a Python process *outside* the container. It connects to the `tmux` session running *inside* the container.
3.  **Observation (Terminal to Text):** The agent captures the current state of the terminal. It literally scrapes the text currently visible on the `tmux` screen, along with recent standard output (stdout) and standard error (stderr).
4.  **Prompting (Text to LLM):** The agent takes this raw terminal text, combines it with the overarching task instructions (the prompt), and formats it into a standard chat message. It then sends this message via an API call to the LLM being evaluated.
5.  **Action Generation (LLM to Text):** The LLM processes the prompt and generates a text response. The agent is programmed to look for specific formatting in this response (e.g., text enclosed in a `<command>` or `<keystroke>` XML tag).
6.  **Execution (Text to Terminal):** The agent parses the LLM's response, extracts the intended command, and translates it into literal keystrokes. It sends these keystrokes over the connection to the `tmux` session, effectively "typing" the command and hitting Enter.
7.  **Iteration:** Steps 3-6 repeat. The LLM observes the result of its previous command on the screen, decides on the next step, and continues until it believes the task is complete (often by outputting a specific `<submit>` token) or until it hits a maximum step limit.

## 4.2 The Evaluation Phase (Verification)

Once the inference loop concludes (either because the LLM submitted a final answer, or because it hit a maximum step/time limit), the evaluation phase begins. 

A major flaw in many modern LLM benchmarks is their reliance on "LLM-as-a-judge"—using another model (like GPT-4) to read the output and guess if it's correct. This is subjective and prone to bias. TerminalBench completely avoids this by relying on **deterministic, programmatic verification**.

Here is exactly how the harness evaluates the agent's work:

1.  **Freezing the State:** The moment the agent signals completion, the `harbor` harness freezes the Docker container. The agent is disconnected, meaning it can no longer make changes.
2.  **Injecting the Test Suite:** Every task in TerminalBench comes with a hidden test suite (usually a `run-tests.sh` bash script or a `test_outputs.py` Python script). The harness injects this script into the frozen container.
3.  **Executing the Tests:** The harness runs the test script *inside* the exact same environment where the agent just operated. This is crucial because it allows the test script to check for complex side-effects. The script might:
    *   **Check File System State:** Did the agent create a specific file? Does the file have the correct permissions? Does a generated JSON file match a specific schema?
    *   **Check Network/Service State:** If the task was to configure a web server, the test script will actually send an HTTP `curl` request to `localhost:8080` to see if the server responds with the expected HTML.
    *   **Check Computational Results:** If the task was to train a machine learning model, the test script will load the saved model weights and run an inference pass against a hidden test dataset to ensure it meets a minimum accuracy threshold.
4.  **Binary Scoring:** The test script is designed to output a strict binary result: `Pass` (exit code 0) or `Fail` (any non-zero exit code). There is no partial credit.
5.  **Teardown and Logging:** Finally, the harness records the score, saves the full trajectory of the agent's actions (for later debugging or error analysis), and destroys the Docker container to ensure a clean slate for the next task.

## 4.3 A Concrete Example: The "New Encrypt Command" Task

Let's look at a practical example to see this loop in action.

*   **The Task:** The agent is placed in a directory with a plaintext file (`secret.txt`) and a custom, undocumented binary executable called `my_encryptor`. The goal is to encrypt the file using a specific password.
*   **The Inference Process:**
    *   *Step 1 (Observation):* The agent scrapes the empty terminal screen and sends it to the LLM along with the prompt: "Encrypt `secret.txt` using `my_encryptor`...".
    *   *Step 2 (Action):* The LLM decides it needs to see what files are present. It outputs `<command>ls -la</command>`. The agent parses this and types `ls -la\n` into `tmux`.
    *   *Step 3 (Observation):* The terminal outputs the directory contents. The agent scrapes this new screen and sends it back to the LLM.
    *   *Step 4 (Action):* Seeing the binary, the LLM realizes it doesn't know the syntax. It outputs `<command>./my_encryptor --help</command>`. The agent types it in.
    *   *Step 5 (Observation):* The terminal outputs the help menu: `Usage: my_encryptor -in <file> -out <file> -pass <password>`. The agent sends this text to the LLM.
    *   *Step 6 (Action):* The LLM reads the help menu, formulates the correct command, and outputs `<command>./my_encryptor -in secret.txt -out encrypted.bin -pass supersecret</command>`. The agent executes it.
    *   *Step 7 (Action):* The LLM outputs a `<submit>` token, signaling to the agent that it believes the task is done. The agent terminates the loop.
*   **The Evaluation:** The `harbor` harness takes over and runs the verification script. The script uses a known decryption tool to decrypt `encrypted.bin` using the password `supersecret`, and then compares the result byte-for-byte against the original `secret.txt`. If they match, the task is marked as a **Pass**.

This analysis highlights why TerminalBench is so rigorous: it tests not just knowledge, but the agent's ability to explore, read documentation on the fly, execute commands, and verify its own work before submitting.


# 5. Takeaways and future directions

## 5.1 What TerminalBench tells us

TerminalBench clearly demonstrates that while LLMs are excellent at generating text and code, acting as autonomous agents in dynamic, stateful environments like the Linux terminal is a different beast entirely. The sub-65% success rate of frontier models highlights the gap between knowing *what* command to run and successfully executing a long-horizon workflow.

## 5.2 Future roadmap

The team behind TerminalBench has an ambitious roadmap. We can expect to see:
*   **More specialized task subsets:** Focusing on areas like web development, personal assistant tasks, and interactive environments (like `vim` or terminal games).
*   **Broader benchmark integration:** Adapters to run other popular benchmarks (like SWE-Lancer and MLE-Bench) through the TerminalBench harness.
*   **Continued difficulty scaling:** As agents improve, TerminalBench will likely introduce even harder tasks to keep pushing the boundaries of what AI can achieve in the CLI.

TerminalBench is poised to become a standard yardstick for agentic AI, pushing developers to build systems that aren't just smart, but practically useful in the environments where real work gets done.

# 6. References

1. **TerminalBench Website:** [tbench.ai](https://www.tbench.ai/)
2. **TerminalBench GitHub Repository:** [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench)
3. **TerminalBench 2.0 Paper (ICLR 2026):** "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces" [OpenReview](https://openreview.net/forum?id=a7Qa4CcHak)
