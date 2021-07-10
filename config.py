###
#
# Word Embedding Evaluation configuration file
#
###
import os
from pathlib import Path

run_on_myriad = False

home = str(Path.home())
dir_name = '/word_embedding_evaluation/'
if run_on_myriad: dir_name = '/Scratch' + dir_name

root_dir = home + dir_name

parameters = {}

parameters['config_file'] = root_dir + 'config.py'

general_data_dir = home + '/data/'
parameters['data_dir'] = os.path.abspath(root_dir + 'data/') + '/'
word_embeddings_dir = os.path.abspath(general_data_dir + 'word_embeddings/') + '/'

# BNC DATA
parameters['bnc_texts_dir'] = general_data_dir + 'British_National_Corpus/Texts/'

bnc_data_name = 'bnc_full_proc_data'
# parameters['bnc_data_name'] = 'bnc_baby_proc_data'
parameters['bnc_data_dir'] = os.path.abspath(general_data_dir + 'British_National_Corpus/bnc_full_processed_data/') + '/'
parameters['bnc_data'] = parameters['bnc_data_dir'] + bnc_data_name + '.txt'

parameters['use_data_subset'] = True
parameters['data_subset_size'] = 0.1
bnc_subset_data_name = bnc_data_name + '_shffl_sub-' + str(parameters['data_subset_size']).strip("0").strip(".")
parameters['bnc_subset_data'] = parameters['bnc_data_dir'] + bnc_subset_data_name + '.txt'
parameters['bnc_subset_tags'] = parameters['bnc_data_dir'] + bnc_subset_data_name + '_tags.txt'

data_name = bnc_subset_data_name if parameters['use_data_subset'] else bnc_data_name

parameters['tokenised_data'] = parameters['data_dir'] + 'tok_' + data_name + '.npy'
parameters['counts_file'] = parameters['data_dir'] + 'counts_' + data_name + '.csv'

parameters['vocab_threshold'] = 5

embeds_dir = parameters['data_dir'] + 'embeds/'

# Word2Vec Embeddings
parameters['word2vec_embeds'] = embeds_dir + 'word2vec_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.npy'
parameters['word2vec_embeds_spacy'] = embeds_dir + 'word2vec_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.txt'

# HellingerPCA Embeddings
parameters['hellingerPCA_embeds'] = embeds_dir + 'hellingerPCA_200d_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.npy'
parameters['hellingerPCA_embeds_spacy'] = embeds_dir + 'hellingerPCA_200d_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.txt'


parameters['source_hellingerPCA_vocab'] = word_embeddings_dir + 'HellingerPCA/vocab.txt'
parameters['source_hellingerPCA_vecs'] = word_embeddings_dir + 'HellingerPCA/words.txt'

# GloVe Embeddings
parameters['glove_embeds'] = embeds_dir + 'glove_wiki-gigaword_300d_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.npy'
parameters['glove_embeds_spacy'] = embeds_dir + 'glove_wiki-gigaword_300d_' + bnc_data_name + '_voc_' + str(parameters['vocab_threshold']) + '.txt'


# parameters['source_glove_embeds'] = word_embeddings_dir + 'GloVe/glove.840B.300d.txt'

# Linguistic feature vectors
linguistic_feats_dir = parameters['data_dir'] + 'linguistic_feats/'
parameters['liwc_features'] = linguistic_feats_dir + 'pennebaker.dic'
parameters['semcor_features_dir'] = linguistic_feats_dir + 'SEMCATdataset2018/Categories/'
parameters['word_features'] = word_embeddings_dir + 'morph_liwc_wordnet_word_features.npy'


# Correlation Data
correlation_dir = parameters['data_dir'] + 'correl_data/'
parameters['simlex_file'] = correlation_dir + 'SimLex-999.txt'
parameters['wordsim353_sim_file'] = correlation_dir + 'wordsim_similarity_goldstandard.txt'
parameters['wordsim353_rel_file'] = correlation_dir + 'wordsim_relatedness_goldstandard.txt'

correlation_results_dir = correlation_dir + 'results/'
parameters['simlex_results_file'] = correlation_results_dir + 'simlex_results.npy'
parameters['wordsim353_sim_results_file'] = correlation_results_dir + 'wordsim353_sim_results.npy'
parameters['wordsim353_rel_results_file'] = correlation_results_dir + 'wordsim353_rel_results.npy'
