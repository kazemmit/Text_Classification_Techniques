from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def Traditional_Feature_Extraction(AllText,TrainX,TestX):
    # Feature Extraction

    # ****************** Count Vectors as features ******************
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
    count_vect.fit(AllText)

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(TrainX)
    xtest_count =  count_vect.transform(TestX)


    # ****************** TF-IDF Vectors as features (Word level (BOW), Word ngram and Char ngram  ******************
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word',stop_words='english' ,token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(AllText)
    xtrain_tfidf =  tfidf_vect.transform(TrainX)
    xtest_tfidf =  tfidf_vect.transform(TestX)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', stop_words='english',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(AllText)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(TrainX)
    xtest_tfidf_ngram =  tfidf_vect_ngram.transform(TestX)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(AllText)
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(TrainX)
    xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(TestX)

    return xtrain_count,xtest_count,xtrain_tfidf,xtest_tfidf,xtrain_tfidf_ngram,xtest_tfidf_ngram,xtrain_tfidf_ngram_chars,xtest_tfidf_ngram_chars

