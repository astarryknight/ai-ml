---
layout: post
title:  Build a Project
nav_order: 5
estimated_time: 8-16 hours
prereqs:
  - Complete several core lessons or have equivalent experience
outcomes:
  - Scope a small problem with a real user or audience
  - Evaluate a baseline and one meaningful iteration
  - Deploy a final project or document a reproducible local demo when hosting is not appropriate
  - Write a README and resume-ready project description
---

# Build a Project

Build one focused project that someone else can understand, run, and evaluate. The goal is a polished, honest artifact, not a large production system.

## Good project scope

Choose one user, one narrow problem, one primary metric, and one simple baseline. Examples:

- Predict a numeric outcome from a small tabular dataset.
- Classify short text with a clearly defined error tradeoff.
- Cluster items and explain how a user could use the groups.
- Build a small retrieval or code-search assistant over a bounded document set.
- Create a simple interactive model demo with a clear input and output.

Avoid projects that require collecting private data, training a large model from scratch, or building a full production platform.

## Milestones

### Milestone 1: Problem and baseline {#milestone-1}

- One-sentence problem statement.
- Intended user or audience.
- Data source and data-quality notes.
- Success metric and why it fits the task.
- Simple baseline model, rule, or prompt workflow.
- One-page project brief.

### Milestone 2: Prototype and evaluation {#milestone-2}

- Reproducible notebook or application code.
- At least one meaningful evaluation metric or test set.
- One iteration based on evidence.
- Short explanation of what improved and what did not.

### Milestone 3: Deployment and showcase {#milestone-3}

- Deployed demo. If deployment is not appropriate for the project, provide reproducible local run instructions and explain why.
- README with problem, approach, data, evaluation, limitations, and setup.
- Screenshots or a short demo video.
- Two-minute presentation.
- Resume-ready project description.

## Project quality checklist

| Area | What good evidence looks like |
| --- | --- |
| Problem framing | The user, problem, scope, and success measure are clear |
| Data and baseline | The data is explained and a simple comparison exists |
| Working prototype | The main workflow runs from input to output |
| Evaluation and iteration | Results use an appropriate test and motivate one change |
| Sharing | Another person can run or view the project |
| Explanation | The README and demo explain the result and its limitations |

## README template

```markdown
# Project name

## Problem
Who is this for, and what does the project help them do?

## Approach
What data, model, or retrieval workflow did you use?

## Evaluation
What metric or test set did you use? What did you learn?

## Demo and setup
How can another person run or view it?

## Limitations
When should someone avoid trusting the result?

## Next step
What would you improve with more time or better data?
```
