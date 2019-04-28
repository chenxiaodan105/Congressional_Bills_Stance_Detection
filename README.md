# Xiaodan's Directory

## Contents
[1. Text Preprocessing](#Text-Preprocessing)

[2. Imbalance Data Handling](#Imbalance-Data-Handling)

[3. Methods](#Methods)

[4. Grid Search](#Grid-Search)


## Text Preprocessing
* tokenization
```
def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token.strip() for token in tokens]
    return tokens
```
* remove punctuation
```
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
```
* remove words that are not purely alphabetic words
```
def remove_non_alphabetic_characters(text):
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if w.isalpha()]
    return ' '.join(tokens)
```
* remove stopwords
```
def remove_stopwords(text):
    stop = list(set(stopwords.words('english')))
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stop]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
```
* remove all words that have a length <= 1 characters (this number can be changed)
```
def remove_tokens_with_frequency(text,count):
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if len(w)>count]
    return ' '.join(tokens)
```
* stemming and lemmatization

## Imbalance Data Handling
* smote 
* balanced ensemble method using SVM or decision tree as base model
* [reference](https://imbalanced-learn.org/en/stable/ensemble.html)
* [reference](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)


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







