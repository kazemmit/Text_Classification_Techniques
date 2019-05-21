from sklearn import svm
from SNN import create_model_architecture_snn
from LSTM import create_rnn_lstm
from BiRNN import create_bidirectional_rnn
from RCNN import create_rcnn
from GRU import create_rnn_gru
from CNN import create_cnn
from Parameters import *
from Traditional_Feature_Extraction import Traditional_Feature_Extraction
from ReadDataAndPreProcess import ReadDataAndPreProcess
from Classifier import Classifier_Train_Test
from NLP_Features import NLP_Features

# All parameters have been defined already in the Parameters.py file

# ****************** Read the samples from a text file, and split them into train and test sets ******************

AllSamplesX,AllSamplesY, TrainX,TrainY,TestX,TestY=ReadDataAndPreProcess(corpus_file,test_size)

# ****************** Traditional Features (BOW, TF, TFIDF, Word-Ngram, Char-Ngram) ******************

xtrain_count,xtest_count,\
xtrain_tfidf,xtest_tfidf,\
xtrain_tfidf_ngram,xtest_tfidf_ngram,\
xtrain_tfidf_ngram_chars,xtest_tfidf_ngram_chars = Traditional_Feature_Extraction(AllSamplesX,TrainX,TestX)

# print(Classifier_Train_Test(svm.SVC(),xtrain_count,TrainY,xtest_count,TestY))
# print(Classifier_Train_Test(svm.SVC(),xtrain_tfidf,TrainY,xtest_tfidf,TestY))
# print(Classifier_Train_Test(svm.SVC(),xtrain_tfidf_ngram,TrainY,xtest_tfidf_ngram,TestY))
# print(Classifier_Train_Test(svm.SVC(),xtrain_tfidf_ngram_chars,TrainY,xtest_tfidf_ngram_chars,TestY))


# ****************** Word Embeddings - Pre-train models [Trained on huge data set such as Wiki - We call them universal] ******************
# Universal Sentence Encoder

# session, embedded_text, text_input = init_USE_model()
# x_train_use_universal = session.run(embedded_text, feed_dict={text_input: TrainX})
# x_test_use_universal = session.run(embedded_text, feed_dict={text_input: TestX})

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

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings", accuracy)