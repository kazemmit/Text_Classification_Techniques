from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset
data = open('corpus.txt').read()
labels, texts = [], []
for line in data.split("\n"):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# create a dataframe using texts and lables
AlldataDF = pandas.DataFrame()
AlldataDF['text'] = texts
AlldataDF['label'] = labels

TrainX,TestX,TrainY,TestY = model_selection.train_test_split(AlldataDF['text'],AlldataDF['label'],test_size=0.1)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
encoder.fit(AlldataDF['label'])
train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)


# Feature Extraction

# ****************** Count Vectors as features ******************
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
count_vect.fit(AlldataDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(TrainX)
xvalid_count =  count_vect.transform(TestX)


# ****************** TF-IDF Vectors as features (Word level (BOW), Word ngram and Char ngram  ******************
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word',stop_words='english' ,token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(AlldataDF['text'])
xtrain_tfidf =  tfidf_vect.transform(TrainX)
xvalid_tfidf =  tfidf_vect.transform(TestX)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', stop_words='english',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(AlldataDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(TrainX)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(TestX)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(AlldataDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(TrainX)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(TestX)


# ****************** Word Embeddings ******************

