# Xiaodan's Directory

## Contents
[0. How to Use](#How-to-Use)

[1. Introduction](#Introduction)
Companies are sensitive to political policies. However, it could bring them a lot of benefits, for example, the fund and support from the government, if they could catch up with political trends beforehand. 

Stance detection, also known as holding attitudes over things, could be the key to political trends.

In this project, NLP method is used to do the prediction based on 2 sessions of congressional bills and transcripts.

Just explore it and have fun !

[2. Text Preprocessing](#Text-Preprocessing)

[3. Imbalance Data Handling](#Imbalance-Data-Handling)

[4. Methods](#Methods)

[5. Grid Search](#Grid-Search)

## How to Use
* The data and pretrained GloVe file's size is beyond the GitHub's limit, so you could download them through links below
* Download [Data](https://drive.google.com/open?id=1wChMSMjrJ9Cbb8wzN3hU8X3fxzA_SakT) 
* Download [pretrained GloVe](https://drive.google.com/open?id=1Ez9jDgCfU4Nar2wobzic79UAGFLnzmHF) for word embedding
* Go to code file and explore!

| file |  use | explaination |
| ----------- | ----------- | ----------- | 
| data_preprocessing.py | text preprocessing | tokenization, split data, remove stop words, remove special words and so on |
| relabel_model.py | relabel data using regular expression | relabel speeches by detecting key words |
| DL_Models.ipython | Deep Learning Models | two models |
| DL_Models_with_F1_score.ipython | Deep learning Models with the metric F1 score||
| ML_Models_with_Imbalance_Data_Handling.ipython  | ML Models and Imbalance data Handling| six models |
| data_relabelling_and_wrangling.ipython |EDA|EDA|

## Introduction


## Text Preprocessing
* tokenization
* remove punctuation
* remove words that are not purely alphabetic words
* remove stopwords
* remove all words that have a length <= 1 characters (this number can be changed)
* limit tokens frequency

## Imbalance Data Handling
* smote 
* balanced ensemble method
* [reference](https://imbalanced-learn.org/en/stable/ensemble.html)
* [paper](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)


## Methods
* stance detection

| method |  F1 score(have stance) | F1 score(no stance) |
| ----------- | ----------- | ----------- | 
| tfidf + lr | 0.21 | 0.85 |
| tfidf + random forest |0.16 | 0.96 |
| tfidf + BalancedRandomForestClassifier |0.22 | 0.83|
| tfidf + WeightedBalancedRandomForestClassifier | 0.19|0.78|
| relabeling model + tfidf + lr  | 0.80| 0.83 |
| relabeling model + tfidf + random forest |0.88|0.91|
| relabeling model + Glove + LSTM |0.86 | 0.96 |
| relabeling model + Glove + CNN + LSTM | 0.86| 0.95 |


## Grid Search
* grid search for tfidf, including ngram, max_df, min_df, stop_words, max_features, etc.
* grid search for machine learning models(LR, random forest)







