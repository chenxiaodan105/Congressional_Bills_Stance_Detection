# Xiaodan's Directory

## Contents
[0. How to Use](#How-to-Use)

[1. Text Preprocessing](#Text-Preprocessing)

[2. Imbalance Data Handling](#Imbalance-Data-Handling)

[3. Methods](#Methods)

[4. Grid Search](#Grid-Search)


## Text Preprocessing
* tokenization
* remove punctuation
* remove words that are not purely alphabetic words
* remove stopwords
* remove all words that have a length <= 1 characters (this number can be changed)

## Imbalance Data Handling
* smote 
* balanced ensemble method
* [reference](https://imbalanced-learn.org/en/stable/ensemble.html)
* [paper](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)


## Methods
* stance detection

| method |  F1 score(majority) | F1 score(minority) |
| ----------- | ----------- | ----------- | 
| tfidf + lr |  |
| tfidf + random forest | |
| tfidf + BalancedRandomForestClassifier | |
| tfidf + WeightedBalancedRandomForestClassifier | |
| relabeling model + tfidf + lr  | |
| relabeling model + tfidf + random forest | |
| Glove + LSTM | |
| Glove + CNN + LSTM | |


## Grid Search
* grid search for tfidf, including ngram, max_df, min_df, stop_words, max_features, etc.
* grid search for machine learning models(LR, random forest)







