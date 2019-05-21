# Pre-train Universal Sentence Encoder URL

use_pretrain_model_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'

# Corpus file's address and name - This file has all samples
# Each line of this file corresponds to one sample, and it should represent both sample's label and sample's text [with space in between]
# For example the first sample of a corpus could be like below:"
# -1 today is a rainy day

corpus_file = 'corpus.txt'

# The ratio of all data that should be used for testing purpose

test_size = 0.1 # means 10% of data to be used for testing purpose
