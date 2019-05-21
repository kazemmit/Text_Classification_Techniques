import pandas,string,textblob

def NLP_Features(TrainX,TestX):
    Alldata = TrainX+TestX
    AlldataDF = pandas.DataFrame()
    AlldataDF['text'] = Alldata

    AlldataDF['char_count'] = AlldataDF['text'].apply(len)
    AlldataDF['word_count'] = AlldataDF['text'].apply(lambda x: len(x.split()))
    AlldataDF['word_density'] = AlldataDF['char_count'] / (AlldataDF['word_count']+1)
    AlldataDF['punctuation_count'] = AlldataDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    AlldataDF['title_word_count'] = AlldataDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    AlldataDF['upper_case_word_count'] = AlldataDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

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

    AlldataDF['noun_count'] = AlldataDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
    AlldataDF['verb_count'] = AlldataDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
    AlldataDF['adj_count'] = AlldataDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
    AlldataDF['adv_count'] = AlldataDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
    AlldataDF['pron_count'] = AlldataDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

    return AlldataDF