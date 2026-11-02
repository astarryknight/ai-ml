---
layout: post
title: Final Project Tracks
nav_exclude: true
estimated_time: 90 minutes in class, then 2-6 hours to adapt
prereqs:
  - Complete the Week 7 model-choice lab or have equivalent experience
outcomes:
  - Choose a project track without starting from a blank page
  - Reproduce a complete pipeline using a public Kaggle dataset
  - Explain the split, preprocessing, baseline, model, metric, and limitations
  - Make one evidence-based adaptation for Week 9
---

# Week 8: Start a final project from a working example

You do not need to invent a workflow from nothing. Choose one track, run its complete reference notebook, understand the important decisions, and then reuse that structure with a dataset you choose.

{: .important }
> **The completed notebooks are examples, not final-project submissions.** Use them to choose a type of project and learn the pipeline. Your project should use a **different dataset**—ideally another public Kaggle dataset—and should update the target or output, features, preprocessing, evaluation, and explanation to match that data. Do not submit the reference notebook unchanged.

## Finish line for this week

By the end of Week 8, you should have:

1. selected one track;
2. run its reference notebook from beginning to end;
3. selected a different dataset for your own project;
4. explained how its target or output, split, preprocessing, baseline, and metric should change; and
5. written down the first adaptation to implement in Week 9.

If you are unsure, choose **Track 1: Predict bike demand**. It is the most direct continuation of the earlier course material.

## Choose a track

| If you want to work with… | Choose | What you will show |
| --- | --- | --- |
| Rows, dates, weather, and numbers | Track 1 · Predict demand | A forecast plotted against actual demand |
| Songs and similarity | Track 2 · Recommend songs | Five recommendations for a selected song |
| Language and classification | Track 3 · Classify text | Predictions, a confusion matrix, and error examples |
| Images and transfer learning | Track 4 · Recognize scenes | Labeled image predictions and per-class results |

## The common project pattern

Every notebook follows the same process. **The process is the reusable part; the example dataset is not.**

1. **Frame:** Who uses the result, and what question does it support?
2. **Acquire:** Download a public Kaggle dataset and record its source.
3. **Inspect:** Understand rows, columns or labels, missing values, and possible leakage.
4. **Split:** Reserve evaluation data before learning transformations or model parameters.
5. **Prepare:** Make cleaning and feature transformations repeatable.
6. **Baseline:** Measure a simple starting approach.
7. **Model:** Fit one model that matches the data and goal.
8. **Evaluate:** Use an appropriate metric and inspect real examples of errors.
9. **Adapt:** Change one decision and compare it with the reference.
10. **Share:** Present one prediction, recommendation, graph, table, or demo with a short explanation.

## Track 1: Predict bike demand

**Question:** How many bikes will people rent in a given hour?

This track uses the real [Bike Sharing in Washington, D.C. dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset). It is a regression project with a chronological split, a median baseline, a preprocessing pipeline, and a random-forest model.

Important decisions:

- split by time rather than randomly because the intended use is forecasting;
- remove `casual` and `registered` because they add up to the target and leak the answer;
- one-hot encode calendar and weather codes because they are categories;
- leave numeric columns unscaled because a random forest uses threshold-based splits;
- compare MAE and RMSE with the median baseline.

<a class="material-action material-action--primary" href="https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-08-final-project-tracks/track-1-bike-demand.ipynb">Open Track 1 in Colab <span aria-hidden="true">↗</span></a>
<a class="material-action" href="../notebooks/week-08-final-project-tracks/track-1-bike-demand.ipynb" download>Download notebook <span aria-hidden="true">↓</span></a>

Your project: find another time-based regression dataset—such as energy use, traffic, sales, weather, or attendance—and adapt the chronological pipeline to it.

## Track 2: Recommend similar songs

**Question:** Which tracks are most similar to a selected song?

This track uses the public [Spotify Tracks Dataset](https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset). It cleans track records, scales eight audio features, and finds nearest neighbors.

Important decisions:

- remove duplicate tracks and rows without the required audio features;
- define similarity using sound-related features rather than popularity;
- standardize features because distance is sensitive to units;
- compare neighbor distances with random recommendations;
- treat feature similarity as a proxy, not proof that someone will like a song.

<a class="material-action material-action--primary" href="https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-08-final-project-tracks/track-2-spotify-recommender.ipynb">Open Track 2 in Colab <span aria-hidden="true">↗</span></a>
<a class="material-action" href="../notebooks/week-08-final-project-tracks/track-2-spotify-recommender.ipynb" download>Download notebook <span aria-hidden="true">↓</span></a>

Your project: choose a different item dataset—such as movies, books, recipes, games, products, or courses—and define what similarity should mean for it.

## Track 3: Build an SMS spam filter

**Question:** Should an incoming text message be treated as spam?

This track uses the public [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), a compact dataset of approximately 5,500 English messages labeled as legitimate (`ham`) or spam. The messages come from older UK and Singaporean sources, so the notebook treats this as a useful teaching dataset rather than a complete picture of modern scams.

Important decisions:

- use a stratified split so both labels appear in similar proportions;
- begin with TF-IDF bigrams so the text features remain inspectable;
- compare with a most-frequent-class baseline;
- use class weighting because spam is the less common label;
- compare spam precision and recall because blocking a real message and missing spam have different costs;
- test custom messages and decision thresholds rather than assuming `0.50` is always best.

<a class="material-action material-action--primary" href="https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-08-final-project-tracks/track-3-sms-spam.ipynb">Open Track 3 in Colab <span aria-hidden="true">↗</span></a>
<a class="material-action" href="../notebooks/week-08-final-project-tracks/track-3-sms-spam.ipynb" download>Download notebook <span aria-hidden="true">↓</span></a>

### Are we using an RNN?

Not for the required path. The core example uses **TF-IDF plus logistic regression** because it runs quickly on a CPU, gives a strong baseline, exposes useful phrases, and keeps the preprocessing decisions visible.

The notebook includes a small bidirectional LSTM/RNN as an optional Colab experiment. It uses the same split and F1 metric so students can test whether the added tokenization, sequence length, embedding, and training time actually improve the result. Complexity is not automatically an upgrade.

Your project: choose a different labeled-text dataset—such as reviews, support tickets, headlines, comments, or article categories—and rebuild the pipeline around its error tradeoffs.

## Track 4: Recognize natural scenes

**Question:** Does an image show buildings, forest, glacier, mountain, sea, or street?

This track uses the real [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). It reads folder-based labels and trains a small classifier on top of a frozen MobileNetV2 feature extractor.

Important decisions:

- keep the provided test folder separate;
- create the training and validation split before evaluation;
- resize and normalize every image consistently;
- apply augmentation only to training images;
- begin with a frozen pretrained network rather than training a CNN from scratch;
- inspect per-class errors and example predictions, not only overall accuracy.

This is the intermediate track. A Colab GPU is recommended, and the Kaggle download is approximately 350 MB.

<a class="material-action material-action--primary" href="https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-08-final-project-tracks/track-4-scene-classifier.ipynb">Open Track 4 in Colab <span aria-hidden="true">↗</span></a>
<a class="material-action" href="../notebooks/week-08-final-project-tracks/track-4-scene-classifier.ipynb" download>Download notebook <span aria-hidden="true">↓</span></a>

Your project: choose a different folder-based image dataset—such as plants, animals, food, waste, weather, or manufactured objects—and adapt the transfer-learning pipeline.

## Move from the example to your dataset

Before Week 9, complete this dataset-swap checklist:

1. Find a public dataset that matches the track's data type.
2. Record its source and license.
3. Define what one row, text, item, or image represents.
4. Name the new target or desired output.
5. Identify information that would leak the answer or be unavailable in real use.
6. Choose a split that matches how the project would actually be used.
7. Decide which preprocessing steps still apply and which must change.
8. Define a simple baseline and an appropriate metric.
9. Update one visible output so it uses your data rather than the example data.

The model may remain similar at first. The important work is making defensible data and evaluation decisions for the new dataset.

## Decision record

Copy this into your notebook or project README and complete it as you work:

```markdown
Project track:
Reference example used as a guide:
My new dataset and source:
Intended user:
Prediction question or desired output:
What one row or example represents:
Target or output:
Features or inputs:
Possible leakage:
Split and why it matches the real use:
Preprocessing and why it is needed:
Baseline:
Model:
Main metric and why it fits:
One visible result:
One limitation:
Week 9 adaptation:
```

## What happens next

- **Week 9:** Adapt the reference to a new question or dataset, evaluate it, and make one evidence-based iteration.
- **Week 10:** Package the result as a reproducible notebook or demo, write the README and limitations, and prepare a short showcase explanation.

## Instructor preparation

- Run all four notebooks before class.
- Keep a cached fallback sample in case Kaggle is unavailable.
- Confirm the dataset pages and licenses remain available.
- Warn students about potentially offensive language in the text dataset.
- Encourage Kaggle credentials through Colab Secrets rather than pasted tokens.
- Form track-based groups while requiring each learner to maintain their own notebook and decision record.

