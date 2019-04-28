# Xiaodan's Directory

## Contents
[1. How to Use](#How-to-Use)

[2. Text Preprocessing](#Text-Preprocessing)

[3. Imbalance Data Handling](#Imbalance-Data-Handling)

[4. Methods](#Methods)

[5. Grid Search](#Grid-Search)

## How to Use




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
| Glove + LSTM |0.93 | 0.93 |
| Glove + CNN + LSTM | 0.93 | 0.93 |


## Grid Search
* grid search for tfidf, including ngram, max_df, min_df, stop_words, max_features, etc.
* grid search for machine learning models(LR, random forest)







