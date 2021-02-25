###
#
# Read word2vec vocab into a file
#
###

import gensim.downloader as api
model = api.load("word2vec-google-news-300")

vocab = model.wv.vocab

vocab = [w for w in vocabulary.keys()]

with open('word2vec_full-vocab_gensim.txt', 'w+') as f:
    f.writelines(["%s\n" % item  for item in vocab])