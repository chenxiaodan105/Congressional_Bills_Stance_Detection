# Contents

## Introduction

**Stance detection** is the extraction of a people’s reaction to a claim made by a primary actor. It is a core part of a set of approaches to capture political trends. Companies are sensitive to political policies. It could bring them a lot of benefits, such as the fund, the support from the government, if they could catch up with political trends beforehand.

In this project, NLP method is used to do the prediction based on 2 sessions of congressional bills and transcripts.Just explore it and have fun !

* github.io address : [Website](chenxiaodan105.github.io/ds-project)

## How to Use
* The data and pretrained GloVe file's size is beyond the GitHub's limit, so you could **`download`** them through links below
* Download [Data](https://drive.google.com/open?id=1wChMSMjrJ9Cbb8wzN3hU8X3fxzA_SakT) 
* Download [pretrained GloVe](https://drive.google.com/open?id=1Ez9jDgCfU4Nar2wobzic79UAGFLnzmHF) for word embedding
* Go to code file and explore!

| file |  use | explaination |
| ----------- | ----------- | ----------- | 
| `data_preprocessing.py` | text preprocessing | tokenization, split data, remove stop words, remove special words and so on |
| `relabel_model.py` | relabel data using regular expression | relabel speeches by detecting key words |
| DL_Models.ipynb | Deep Learning Models | two models |
| DL_Models_with_F1_score.ipynb | Deep learning Models with the metric F1 score||
| ML_Models.ipynb  | ML Models and Imbalance data Handling| six models |
| cutoff_analysiss_for_speech_length.ipynb |EDA|EDA|






