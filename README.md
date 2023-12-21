# Twitter Emotion Analysis

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/florianehmann/twitter-emotion)
![GitHub top language](https://img.shields.io/github/languages/top/florianehmann/twitter-emotion)

Using a Language Model to Detect Emotion in Tweets

The corresponding [twitter-emotion-webapp](https://github.com/florianehmann/twitter-emotion-webapp) is hosted on [feelthetweet.com](https://www.feelthetweet.com)

## About

In this project I build a model that classifies tweets into the six emotion categories: anger, fear, joy, love, sadness and surprise.

In the first iteration I fine-tuned the [DistilBERT](https://huggingface.co/distilbert-base-uncased) model to classify tweets in the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset, reaching an accuracy of 93.2 % of the validation split of the dataset. This worked reasonably well but was entirely limited to english tweets.

Therefore, I tried a second approach based on the [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) model. This model is pre-trained on 2.5 TB of text data in 100 different languages, giving it an understanding of a wide variety of languages. I then fine-tuned the model on the same dataset of english tweets to build a classifier, this time reaching 92.9 % accuracy on the validation split of the dataset, about as good as the DistilBERT-based approach. The major advantage with the XLM-R-based model, however, is that we get pretty decent results even for non-English languages through a zero-shot cross-lingual transfer.

## Usage

### Setting Up the Environment

To use this project, first set up a conda environment:

```bash
conda env create -f environment.yml
```

although I strongly recommend using `mamba` instead of `conda`.
This environment assumes you have an NVIDIA GPU that can run CUDA 12.1.
If your machine doesn't have a GPU, use

```bash
conda env create -f environment-cpu.yml
```

instead. With this environment, all the computations are done on the CPU and everything works, albeit a lot slower.

### Performing the Fine-Tuning

The fine-tuning of the base XLM-RoBERTa model happens in the notebook `notebooks/XLM-R Fine-Tuning.ipynb` and should take only a couple of minutes on a decent GPU.

### Evaluating the Model

The fine-tuned model is available on [Hugging Face](https://huggingface.co) as [florianehmann/xlm-roberta-base-finetuned-emotion](https://huggingface.co/florianehmann/xlm-roberta-base-finetuned-emotion).
In the notebook `notebooks/XLM-R Results.ipynb`, we evaluate it by creating a confusion matrix for the classification and analyzing some of the misclassifications of the model.
