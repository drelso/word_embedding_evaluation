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
word_features_name = 'morph_liwc_wordnet_word_features_' + bnc_subset_data_name + '_voc_' + str(parameters['vocab_threshold'])
parameters['word_features'] = linguistic_feats_dir + word_features_name + '.npy'
parameters['feature_threshold'] = 5
trimmed_word_features_name = word_features_name + '_featmin_' + str(parameters['feature_threshold'])
parameters['word_feature_vectors'] = linguistic_feats_dir + trimmed_word_features_name + '_vecs.npy'
parameters['word_feature_vectors_spacy'] = linguistic_feats_dir + trimmed_word_features_name + '_vecs.txt'


# Correlation Data
correlation_dir = parameters['data_dir'] + 'correl_data/'
parameters['simlex_file'] = correlation_dir + 'SimLex-999.txt'
parameters['wordsim353_sim_file'] = correlation_dir + 'wordsim_similarity_goldstandard.txt'
parameters['wordsim353_rel_file'] = correlation_dir + 'wordsim_relatedness_goldstandard.txt'
parameters['simverb3500_rel_file'] = correlation_dir + 'SimVerb-3500.txt'

correlation_results_dir = correlation_dir + 'results/'
parameters['simlex_results_file'] = correlation_results_dir + 'simlex_results.npy'
parameters['wordsim353_sim_results_file'] = correlation_results_dir + 'wordsim353_sim_results.npy'
parameters['wordsim353_rel_results_file'] = correlation_results_dir + 'wordsim353_rel_results.npy'
parameters['simverb3500_rel_results_file'] = correlation_results_dir + 'simverb3500_rel_results.npy'


# Word pairs
word_pairs_dir = parameters['data_dir'] + 'word_pairs/'
parameters['train_skipgram_data'] = word_pairs_dir + 'skipgram_bnc_full_proc_data_shffl_sub-1_train.npy'

parameters['num_word_pairs'] = 3000
parameters['word_pair_file'] = word_pairs_dir + str(parameters['num_word_pairs']) + 'rand_ctx_syn_word_pairs.npy'
parameters['word_pair_dists_file'] = word_pairs_dir + str(parameters['num_word_pairs']) + 'rand_ctx_syn_word_pair_dists.npy'

parameters['word_pair_dists_plot'] = word_pairs_dir + str(parameters['num_word_pairs']) + 'rand_ctx_syn_word_pair_dists'