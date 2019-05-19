# Go to this link https://github.com/allenai/allennlp
# Follow Installing via pip (generally conda is already installed on aws servers)
# Copy this file to the server and run it

from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
from operator import add
import math

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

from pymongo import *

conn = MongoClient("mongodb+srv://arpit:groupten123!@agnesdatabase-1mhnx.mongodb.net/Agnes?retryWrites=true")
db = conn.AI_RESEARCH
db_IMDB= db['analogy']

field_name_words ='words' # 'text_CGLOVE_SUBJ'# 'words'
field_name_model = 'CWR' # 'CHAR_CWR'# 'CWR'


docs = db_IMDB.find()
ids = []
counter = 0
whole_had = 0
for doc in docs:
    #if doc['subclass']!=10 :
    #    continue

    whole_had+=1

    # if whole_had <=3800:
    #     continue

    print(whole_had)
    if field_name_model in doc.keys():
        continue

    print(counter)
    counter+=1

    _id = doc['_id']
    sentences = []
    if len(doc[field_name_words])==0:
        db_IMDB.remove({'_id':doc['_id']})
        continue

    sentences.append(doc[field_name_words])
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    message_embedding = embeddings['elmo_representations'][0][0]
    init_model = np.zeros([len(message_embedding[0])], dtype=np.float32)
    words_model = init_model
    word_counter = 0

    for word in message_embedding:
      try:
          word_model = [float(x) for x in word]
          if math.isnan(word_model[0]):
              continue
          words_model = map(add, words_model, word_model)
          word_counter += 1
      except:
          do = 0

    words_model = list(words_model)
    if word_counter == 0:
      word_counter = 1

    final_model = [x / word_counter for x in words_model]

    db_IMDB.update({'_id': _id}, {'$set': {field_name_model: [float(x) for x in final_model]}})




# use batch_to_ids to convert sentences to character ids
#sentences = [['First', 'sentence', '.'], ['Another', '.']]

#print(embeddings)
#print(embeddings[0])
#print(embeddings[1])
#print(len(embeddings))
#print(len(embeddings[0]))
#print(embeddings)


# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector