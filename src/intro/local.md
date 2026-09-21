---
layout: post
title:  Optional Local Setup
parent: Foundations
nav_order: 3
estimated_time: 30-60 minutes
prereqs:
  - A completed Colab notebook
outcomes:
  - Know when local setup is useful
  - Create a Python environment with one tested method
  - Run extension projects that need local packages
---

# Optional Local Setup

## When to use local setup

The core course runs in [Google Colab](https://colab.research.google.com). Use local setup only when you want to work offline, use a local IDE, or complete an extension such as [Snake with RL](../minis/rl.html).

If you are new to the course, skip this page and start with [Foundations](./intro.html).

## Python and an editor

Install [Python](https://www.python.org/downloads/) and use any Python-capable editor. [VS Code](https://code.visualstudio.com/) is one reasonable choice, but it is not required.

## Choose one environment manager

You only need one setup path. Do not install both `pip` and Conda environments for the same project.

### Python virtual environment

```bash
python -m venv aiml
```

Activate it:

```bash
# macOS/Linux
source aiml/bin/activate

# Windows PowerShell
.\aiml\Scripts\Activate.ps1
```

Install the core packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

Deactivate with `deactivate`.

### Conda

If you already use Conda, install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and run:

```bash
conda create -n aiml python=3.11
conda activate aiml
conda install numpy pandas matplotlib scikit-learn jupyter
```

Deactivate with `conda deactivate`.

## Git and GitHub

Git is useful for saving project history and publishing a portfolio. It is optional for Weeks 1–6, but recommended for the capstone. Start with the [Git installation guide](https://git-scm.com/downloads) and the [GitHub documentation](https://docs.github.com/en/get-started).

## Troubleshooting

- If a package import fails in Colab, restart the runtime and rerun the setup cell.
- If a local notebook uses the wrong Python version, select the `aiml` environment as the notebook kernel.
- If a project only works locally, keep a saved sample output in the repository so another student can still understand the expected result.
