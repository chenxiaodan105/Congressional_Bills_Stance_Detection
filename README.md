# Xiaodan's Directory

## Contents
[0. How to Use](#How-to-Use)

[1. Introduction](#Introduction)

[2. Text Preprocessing](#Text-Preprocessing)

[3. Imbalance Data Handling](#Imbalance-Data-Handling)

[4. Methods](#Methods)

[5. Grid Search](#Grid-Search)

## How to Use
Data: [link](https://drive.google.com/open?id=1wChMSMjrJ9Cbb8wzN3hU8X3fxzA_SakT)
GloVe file: [link](https://drive.google.com/open?id=1Ez9jDgCfU4Nar2wobzic79UAGFLnzmHF)


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







