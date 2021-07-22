###
#
# Embedding Evaluation
#
###

## Load embeddings

### BERT embeddings
## Sentence Transformers
## from https://github.com/UKPLab/sentence-transformers
## and paper at https://arxiv.org/pdf/1908.10084.pdf
## pip install -U sentence-transformers

# Full list of models is available in
# pip install spacy-transformers
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# # print(f'dir(tokenizer): {dir(tokenizer)}\n\n')
# print(f'tokenizer("hello"): {tokenizer("hello")}')
# unk_ids = tokenizer("organophosphorus")['input_ids']
# unk_toks = [tokenizer.ids_to_tokens[ix] for ix in unk_ids]
# print(f'\n\ntokenizer("organophosphorus") IDS: {unk_ids}')
# print(f'\n\ntokenizer.ids_to_tokens(unk_toks): {unk_toks}\n\n')
# print(f'MODEL WORD EMBEDDINGS: {model.embeddings.word_embeddings}')

import csv
import os

import numpy as np
import scipy

from config import parameters

from utils.funcs import print_parameters
from utils.word_rep_funcs import build_vocabulary, word2vec_with_vocab, hellingerPCA_with_vocab, glove_with_vocab, word_features, trim_word_features, feature_vectors, get_features_for_word
from utils.word_rep_eval import get_word_pairs, word_pair_distances, plot_word_pair_dists


def read_correl_data(correl_data_path, correl_name='SimLex'):
    with open(correl_data_path, 'r') as f:#, \
        # open(save_file, 'w+') as s:
        data = csv.reader(f, delimiter='\t')
        
        # if re.search('SimLex', correl_name):
        if correl_name == 'SimLex':
            # SimLex dataset has a header
            header = next(data)
            score_index = header.index('SimLex999')
        elif correl_name == 'SimVerb-3500':
            # SimVerb-3500 dataset scores
            # appear in the fourth column
            score_index = 3
        else:
            # WordSim353 datasets scores
            # appear in the third column
            score_index = 2
        valid_pairs = 0
        total_pairs = 0
        # results = [col_names]
        
        correl_data = []

        for row in data:
            word_1 = row[0].lower()
            word_2 = row[1].lower()
            score = row[score_index]
            
            correl_data.append({'w1': word_1, 'w2': word_2, 'score': score})

        return correl_data



def correl_dists_for_embeds(embed_dict, correl_data_path, correl_save_file, correl_name):#, dist_metric='cos'):
    if not os.path.exists(correl_save_file):
        print(f'No word pair distances file for {correl_name} found at {correl_save_file}, calculating distances.')
        
        correl_data = read_correl_data(correl_data_path, correl_name=correl_name)

        correl_results = {
            'w1': [],
            'w2': [],
            'score': []
        }

        first_embeds = True

        for embed_name, embed_file in embed_dict.items():
            print(f"Processing correlation distances for {embed_name} embeddings", flush=True)
            correl_results[embed_name + '_cos'] = []
            correl_results[embed_name + '_euc'] = []

            for word_pair in correl_data:
                if first_embeds:
                    correl_results['w1'].append(word_pair['w1'])
                    correl_results['w2'].append(word_pair['w2'])
                    correl_results['score'].append(float(word_pair['score']))
                
                embeds = np.load(embed_file, allow_pickle=True)
                if word_pair['w1'] in embeds.item().keys() \
                    and word_pair['w2'] in embeds.item().keys():
                    emb_1 = embeds.item()[word_pair['w1']]
                    emb_2 = embeds.item()[word_pair['w2']]

                    cos_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)
                    euc_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                else:
                    cos_dist = np.nan
                    euc_dist = np.nan
                correl_results[embed_name + '_cos'].append(cos_dist)
                correl_results[embed_name + '_euc'].append(euc_dist)
            
            first_embeds = False
        
        print(f"Saving word pair scores and distances ({len(correl_results['w1'])} rows) to {correl_save_file}")
        np.save(correl_save_file, correl_results)
    else:
        print(f'Word pair distances file found at {correl_save_file}.')
        
    

def calculate_correl_results(correl_dists_file, correl_results_save_file=None):
    if correl_results_save_file and not os.path.exists(correl_results_save_file):
        print(f'No correlation results file found at {correl_results_save_file}, calculating results.')
        print(f"Calculating correlation results for distances in {correl_dists_file}")
        correl_dists = np.load(correl_dists_file, allow_pickle=True)
        embed_names = [key for key in correl_dists.item().keys() if key not in ['w1', 'w2', 'score']]
        
        correl_results = {}

        for embed_name in embed_names:
            print(f"Calculating correlation results for {embed_name} embeddings", flush=True)

            scores = []
            dists = []
            num_missing = 0
            for i, dist in enumerate(correl_dists.item()[embed_name]):
                if not np.isnan(dist):
                    scores.append(correl_dists.item()['score'][i])
                    dists.append(correl_dists.item()[embed_name][i])
                else:
                    num_missing += 1

            print(f"\tSkipping {num_missing} words not in vocabulary")

            spearman = scipy.stats.spearmanr(scores, dists)
            pearson = scipy.stats.pearsonr(scores, dists)
            correl_results[embed_name] = {'spearman': spearman, 'pearson': pearson}
        
        print(f"Correlation results: \n\n{correl_results}")

        if correl_results_save_file:
            print(f"Saving correlation results to {correl_results_save_file}")
            np.save(correl_results_save_file, correl_results)
    else:
        print(f'Correlation results file found at {correl_results_save_file}.')
        



if __name__ == "__main__":
    print_parameters(parameters)

    vocabulary = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_threshold'])
    
    word2vec_with_vocab(vocabulary, parameters['word2vec_embeds'])

    hellingerPCA_with_vocab(
        vocabulary,
        parameters['source_hellingerPCA_vocab'],
        parameters['source_hellingerPCA_vecs'],
        parameters['hellingerPCA_embeds'])
    
    glove_with_vocab(
        vocabulary,
        parameters['glove_embeds'])

    word_features(
        vocabulary,
        parameters['liwc_features'],
        parameters['word_features'])
    
    trimmed_feat_dict = trim_word_features(
                            parameters['word_features'],
                            parameters['feature_threshold'])

    feature_vectors(
        vocabulary,
        trimmed_feat_dict,
        parameters['word_feature_vectors'],
        spacy_feats_save_file=parameters['word_feature_vectors_spacy'],
        sort_feats=True)

    # Word pairs
    get_word_pairs(
        parameters['num_word_pairs'],
        vocabulary,
        parameters['train_skipgram_data'],
        parameters['word_pair_file'])

    embed_files_dict = {
        'Word2Vec' : parameters['word2vec_embeds'],
        'HellingerPCA' : parameters['hellingerPCA_embeds'],
        'GloVe' : parameters['glove_embeds'],
        'Feature_Vectors' : parameters['word_feature_vectors']
    }

    word_pair_distances(
        parameters['word_pair_file'],
        embed_files_dict,
        parameters['word_pair_dists_file'])

    plot_word_pair_dists(
        parameters['word_pair_dists_file'],
        parameters['word_pair_dists_plot'] + '_cos',
        fig_title='Word Pair Cosine Distance Distributions',
        dist_type='cos')
    
    plot_word_pair_dists(
        parameters['word_pair_dists_file'],
        parameters['word_pair_dists_plot'] + '_euc',
        fig_title='Word Pair Euclidean Distance Distributions',
        dist_type='euc')

    # Sample word features
    # Takes a while due to the loading of feature dictionary
    # word = 'organic'
    # get_features_for_word(word, parameters['word_feature_vectors'])

    # word_liwc_feats = get_liwc_vectors(parameters['liwc_features'], parameters['word_features'])

    # simlex_data = read_correl_data(parameters['simlex_file'], correl_name='SimLex')
    # wordsim_sim_data = read_correl_data(parameters['wordsim35_sim_file'], correl_name='WordSim353_Sim')
    # wordsim_rel_data = read_correl_data(parameters['wordsim35_rel_file'], correl_name='WordSim_Rel')

    # print(simlex_data)

    # correls_dict = {
    #     'SimLex':           parameters['simlex_results_file'],
    #     'WordSim353-sim':   parameters['wordsim353_sim_results_file'],
    #     'WordSim353-rel':   parameters['wordsim353_rel_results_file'],
    # }

    
    ## Similarity-Distance Correlation
    embeds_dict = {
        'word2vec':         parameters['word2vec_embeds'],
        'GloVe':            parameters['glove_embeds'],
        'hellingerPCA':     parameters['hellingerPCA_embeds'],
        'Feature_Vecs':     parameters['word_feature_vectors']
    }

    correl_dists_for_embeds(
        embeds_dict,
        parameters['simlex_file'],
        parameters['simlex_dists_file'],
        'SimLex-999')
    
    calculate_correl_results(
        parameters['simlex_dists_file'],
        correl_results_save_file = parameters['simlex_results_file'])

    correl_dists_for_embeds(
        embeds_dict,
        parameters['wordsim353_sim_file'],
        parameters['wordsim353_sim_dists_file'],
        'WordSim353-sim')

    calculate_correl_results(
        parameters['wordsim353_sim_dists_file'],
        correl_results_save_file = parameters['wordsim353_sim_results_file'])
    
    correl_dists_for_embeds(
        embeds_dict,
        parameters['wordsim353_rel_file'],
        parameters['wordsim353_rel_dists_file'],
        'WordSim353-rel')

    calculate_correl_results(
        parameters['wordsim353_rel_dists_file'],
        correl_results_save_file = parameters['wordsim353_rel_results_file'])
    
    correl_dists_for_embeds(
        embeds_dict,
        parameters['simverb3500_rel_file'],
        parameters['simverb3500_rel_dists_file'],
        'SimVerb-3500')
    
    calculate_correl_results(
        parameters['simverb3500_dists_file'],
        correl_results_save_file = parameters['simverb3500_results_file'])

    
    # calculate_correl_results(parameters['wordsim353_sim_dists_file'])
    # calculate_correl_results(parameters['wordsim353_rel_dists_file'])
    # calculate_correl_results(parameters['simverb3500_rel_dists_file'])
    