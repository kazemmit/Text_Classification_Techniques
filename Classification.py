from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble

import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from Universal_Sentence_Encoder import init_USE_model
from SNN import create_model_architecture_snn
from LSTM import create_rnn_lstm
from BiRNN import create_bidirectional_rnn
from RCNN import create_rcnn
from GRU import create_rnn_gru


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


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
train_y = encoder.transform(TrainY)
valid_y = encoder.transform(TestY)



# ****************** Word Embeddings - Pre-train models [Trained on huge data set such as Wiki - We call them universal] ******************
# Universal Sentence Encoder

session, embedded_text, text_input = init_USE_model()
x_train_use_universal = session.run(embedded_text, feed_dict={text_input: TrainX})
x_test_use_universal = session.run(embedded_text, feed_dict={text_input: TestX})

# Elmo (Contextualized Word Representations)
pass

# FastText
pass

# Word2vec
pass

# GloVe
pass

# ****************** Word Embeddings - Trained on Amazon Reviews Dataset models [We call them local] ******************

# FastText
pass

# Word2vec
pass

# GloVe
pass

# C-Glove [Character based GloVe]


# ****************** NLP based features ******************

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))


classifier = create_model_architecture_snn(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print ("Shallow NN, Ngram Level TF IDF Vectors:", accuracy)


classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-LSTM, Word Embeddings:", accuracy)

classifier = create_rcnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings", accuracy)

classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-Bidirectional, Word Embeddings", accuracy)

classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-GRU, Word Embeddings", accuracy)