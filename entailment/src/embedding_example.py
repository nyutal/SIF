import sys
import os
sys.path.append('../../src')
import data_io as dio
import multinli_handler as mnh
import sif_word_embedding as swe
import params


multi_nli_file = '../data/multinli_0.9/multinli_0.9_train.txt'
wordfile = '../../../data/glove.6B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightparam = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]

# load word vectors
print('hello')
(words, We) = dio.getWordmap(wordfile) # We = 300d vectors

word2weight = dio.getWordWeight(weightfile, weightparam) # word2weight['str'] is the weight for the word 'str'
weight4ind = dio.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
print('hello2')
# parse nli data base
mlh = mnh.MultiNliHandler()

x, m = mlh.get_all_senteces(multi_nli_file, words)
#x1, m1, x2, m2, golds = mlh.parse_train_data(multi_nli_file, words)
print('hello3')
print(x[:10])
# parameters
params = params.params()
params.rmpc = 1 #rmpc
w = dio.seq2weight(x, m, weight4ind) # get word weights
sif_words = swe.sif_word_embedding(We, x, w, params)
print('hello4')
swe.write_word_embedding('./data/sif_embedding_' + os.path.split(wordfile)[1])


