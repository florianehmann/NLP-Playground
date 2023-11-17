# Twitter Emotion Analysis

![Static Badge](https://img.shields.io/badge/project%20type-personal-blue)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/florianehmann/twitter-emotion)
![GitHub top language](https://img.shields.io/github/languages/top/florianehmann/twitter-emotion)

Using a Language Model to Detect Emotion in Tweets

## About

In this project we build a model that classifies tweets into the six emotion categories: anger, fear, joy, love, sadness and surprise.
To do this, we fine-tune the [DistilBERT](https://huggingface.co/distilbert-base-uncased) model to classify tweets in the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## Usage

### Setting Up the Environment

To use this project, first set up a conda environment. If your machine doesn't have a GPU, use

    $ conda env create -f environment-cpu.yml

If your machine has an NVIDIA GPU that can run CUDA 12.1, use

    $ conda env create -f environment-cuda12.1.yml

### Performing the Fine-Tuning

The fine-tuning of the base DistilBERT model happens in the notebook `notebooks/DistilBERT Fine-Tuning.ipynb` and should take only a couple of minutes on a decent GPU.

### Evaluating the Model

The fine-tuned model is available on [Hugging Face](https://huggingface.co) as [florianehmann/distilbert-base-uncased-finetuned-emotion](https://huggingface.co/florianehmann/distilbert-base-uncased-finetuned-emotion).
In the notebook `notebooks/DistilBERT Results.ipynb`, we evaluate it by creating a confusion matrix for the classification and analyzing some of the misclassifications of the model.
