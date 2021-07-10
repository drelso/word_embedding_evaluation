###
#
# Intrinsic Evaluation Metrics
#
###


import time
import numpy as np
import torch

import csv
import re

from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from utils.funcs import npy_to_tsv
from utils.model_paths import get_model_path, get_vocab_for_name, get_vocab_num, get_correl_dict, get_base_embeds, get_word_pair_datasets, get_syn_dists_basedir, get_base_embeds#, which_files_exist
# from utils.spatial import pairwise_dists, explained_variance, get_embedding_norms, calculate_singular_vals
from utils.plot import plot_exp_variance, plot_dists_dict_histogram, plot_histogram, plot_exp_variances, plot_dists_dict_histo_subplot, plot_dists_histogram_csv, word_pair_dists_mean_var, plot_mean_vars
from utils.data_processing import calculate_histograms, calculate_missing_values
from utils.validation import word_pair_distances, word_pair_dists_dict, correl_coefficients_for_vocab




def correl_all_data_min_vocab(min_vocab_num=20, distance='cos', vocab_sizes=[3,7,20], ratio_models=False):#dataset_path, save_file, vocab_path, embs_list, incl_header=True, score_type='similarity', data_has_header=False, score_index=2):
    """
    Read a similarity dataset and keep
    only the examples that appear in the
    vocabulary.
    
    Requirements
    ------------
    import re
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    dataset_path : str
        path to the similarity data to process
    save_file : str
        path to save the correlation data to
    vocab_path : str
        path to the vocabulary file
    embs_list : list[[str,str]]
        list of tuples where the first element
        is the name of the embeddings and the
        second is the path to the word embeddings
        (requires NPY format)
    incl_header : bool, optional
        whether to include a header in
        the file or not (default: True)
    
    NOTE: Make sure all the embeddings on the
          list can fit in memory at the same time
    """
    
    correl_datasets = get_correl_datasets()
    
    correl_results_dir = 'data/validate/correlation/'
    
    min_vocab_path = get_vocab_num(min_vocab_num)
    print('Minimum vocabulary path:', min_vocab_path)
    with open(min_vocab_path, 'r') as v:
        min_vocab = [row[0] for row in csv.reader(v)]
    
    # embs_dict = get_model_path('embeds', ratio_models)
    # print(embs_dict)
    
    base_embs = get_base_embeds()
    print('\nBase embeds: ', base_embs)
    
    print('Vocab length:', len(min_vocab))
    # print('Vocab first 10: ', min_vocab[:10])
    
    for vocab_size in vocab_sizes:
        print('Vocabulary %d' % (vocab_size))
        vocab_path = get_vocab_num(vocab_size)
        with open(vocab_path, 'r') as v:
            # vocab = [row[0] for row in csv.reader(v)]
            vocab = {w[0]: i for i, w in enumerate(csv.reader(v))}
            
        embs_dict = get_embeds_for_vocab(vocab_size, ratio_models)
        print('Models for vocab: ', embs_dict)
        vocab_re = '_v' + str(vocab_size)
        emb_names = [name for name in embs_dict.keys()]
        emb_col_names = [re.sub(vocab_re, '', name) for name in emb_names]
        col_names = ['word_1', 'word_2', 'score'] + emb_col_names
        
        embs = {name: np.load(path) for name, path in embs_dict.items()}
        for correl_name, correl_data_path in correl_datasets.items():
            print('%s correl dataset path: \t %s' % (correl_name, correl_data_path))
            
            print('Col names: ', col_names)
            if not ratio_models:
                save_file = correl_results_dir + correl_name + '_v' + str(vocab_size) + '.csv'
            else:
                save_file = correl_results_dir + correl_name + '_ratios_v' + str(vocab_size) + '.csv'
            
            with open(correl_data_path, 'r') as f, \
                open(save_file, 'w+') as s:
                data = csv.reader(f, delimiter='\t')
                writer = csv.writer(s)
                if re.search('SimLex', correl_name):
                    # SimLex dataset has a header
                    header = next(data)
                    score_index = header.index('SimLex999')
                else:
                    # WordSim353 datasets scores
                    # appear in the third column
                    score_index = 2
                valid_pairs = 0
                total_pairs = 0
                results = [col_names]
                
                for row in data:
                    word_1 = row[0].lower()
                    word_2 = row[1].lower()
                    score = row[score_index]
                    
                    
                    if word_1 in min_vocab and word_2 in min_vocab:
                        temp_row = [None] * len(col_names)
                        temp_row[col_names.index('word_1')] = word_1
                        temp_row[col_names.index('word_2')] = word_2
                        temp_row[col_names.index('score')] = score
                        valid_pairs += 1
                        
                        for emb_name in emb_names:
                            emb_1 = embs[emb_name][vocab[word_1]]
                            emb_2 = embs[emb_name][vocab[word_2]]
                            if distance == 'cos':
                                dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                            elif distance == 'euc':
                                dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                            else:
                                raise ValueError('Invalid value for distance metric: %s' % (distance))
                            emb_col = re.sub(vocab_re, '', emb_name)
                            temp_row[col_names.index(emb_col)] = dist
                        
                        results.append(temp_row)
                    total_pairs += 1
                
                print('Valid word pairs in %s: %d/%d' % (correl_name, valid_pairs, total_pairs))
                print('Writing to ', save_file)
                writer.writerows(results)




def read_correl_data(correl_data_path, correl_name='SimLex'):
    with open(correl_data_path, 'r') as f:#, \
        # open(save_file, 'w+') as s:
        data = csv.reader(f, delimiter='\t')
        
        writer = csv.writer(s)
        if re.search('SimLex', correl_name):
            # SimLex dataset has a header
            header = next(data)
            score_index = header.index('SimLex999')
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
        

        #     if word_1 in min_vocab and word_2 in min_vocab:
        #         temp_row = [None] * len(col_names)
        #         temp_row[col_names.index('word_1')] = word_1
        #         temp_row[col_names.index('word_2')] = word_2
        #         temp_row[col_names.index('score')] = score
        #         valid_pairs += 1
                
        #         for emb_name in emb_names:
        #             emb_1 = embs[emb_name][vocab[word_1]]
        #             emb_2 = embs[emb_name][vocab[word_2]]
        #             if distance == 'cos':
        #                 dist = scipy.spatial.distance.cosine(emb_1, emb_2)
        #             elif distance == 'euc':
        #                 dist = scipy.spatial.distance.cosine(emb_1, emb_2)
        #             else:
        #                 raise ValueError('Invalid value for distance metric: %s' % (distance))
        #             emb_col = re.sub(vocab_re, '', emb_name)
        #             temp_row[col_names.index(emb_col)] = dist
                
        #         results.append(temp_row)
        #     total_pairs += 1
        
        # print('Valid word pairs in %s: %d/%d' % (correl_name, valid_pairs, total_pairs))
        # print('Writing to ', save_file)
        # writer.writerows(results)




def correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=False):
    """
    Create a plot from a correlation
    data file in the format of
    correlation_data()
    
    Requirements
    ------------
    import csv
    scipy.stats.pearsonr
    scipy.stats.spearmanr
    correl_paths_for_vocab
    
    Parameters
    ----------
    data_file : str
        path to the correlation data
        file with the following columns:
        - word_1
        - word_2
        - relatedness / similarity
        - cos_w2v
        - euc_w2v
        - cos_w2v_syns
        - euc_w2v_syns
        - cos_rand
        - euc_rand
    score_type : str, optional
        type of score being ploted. Options are:
        - similarity
        - relatedness
        - SimLex999
    """
    correl_result_files = correl_paths_for_vocab(vocab_num, ratio_models)
    print('Correl result files: ', correl_result_files)
    
    full_correl_results = {}
    
    for correl_name, correl_file in correl_result_files.items():
        print('Running %s for vocabulary %d' % (correl_name, vocab_num))
        with open(correl_file, 'r') as f:
            data = csv.reader(f)
            header = next(data)
            
            # Creates a dictionary of column names
            cols = {w: i for i, w in enumerate(header)}
            correls_dict = {name: [] for name in header if name != 'word_1' and name != 'word_2'}
            # print('Columns: ', cols)
            
            for row in data:
                for key in correls_dict.keys():
                    correls_dict[key].append(float(row[cols[key]]))
            
        results = {}
        for model_name in correls_dict.keys():
            if model_name != 'score':
                if correl_metric == 'spearmanr':
                    correl_score = scipy.stats.spearmanr(correls_dict['score'], correls_dict[model_name])
                elif correl_score == 'pearsonr':
                    correl_score = scipy.stats.pearsonr(correls_dict['score'], correls_dict[model_name])
                else:
                    raise ValueError('Correlation metric "%s" not defined.' % (correl_metric))
                results[model_name] = correl_score[0]
                # print('%s for %s: \t %r' % (correl_metric, model_name, correl_score[0]))
        
        full_correl_results[correl_name] = results
    
    return full_correl_results





if __name__ == '__main__':
    
    start_time = time.time()
    
    ratio_models = False

    correlation_data2(correl_dataset_dict, embs_dict, incl_header=True, score_type='similarity', data_has_header=False, score_index=2)

    correl_coeffs = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    print('Correl coeffs:', correl_coeffs)
    
    # Calculate pairwise distance histograms
    # calculate_missing_values('dist_hists', ratio_models=ratio_models)
    '''
    dist_hist_dict = get_model_path('dist_hists', ratio_models=ratio_models)
    save_file = 'plots/dist_hists/w2vi_s_v7_ratios_norm_dists_hist_subp.png'
    # plot_dists_dict_histogram(dist_hist_dict, save_file=save_file, title='W2V init. syns ratios voc-7 distances histogram', model_filter='', norm=True)
    plot_dists_dict_histo_subplot(dist_hist_dict, save_file=save_file, title='W2V init. voc-7 syn. ratios pairwise distances', model_filter='', norm=True)
    '''
    # norms_dict = get_model_path('norms')
    # save_file = 'plots/norm_hists/w2v_init_ns_l2_hist.png'
    # plot_histogram(norms_dict, num_bins=100, save_file=save_file, title='Word2Vec in. no-syns L2 norms hists', model_filter='w2vi_ns', norm=False, x_label='Norms (bins)')
    
    # S_dict = get_model_path('sing_vals')
    # save_file = 'plots/explained_var/w2v_init_s_exp_vals.png'
    # plot_exp_variances(S_dict, save_file=save_file, model_filter='w2vi_s', title='Word2Vec init syns explained variances', log_scale=False, plot=False)
    
    '''
    # CALCULATE CORRELATION SCORES
    ratio_models = False
    vocab_num = 20
    
    # correlation_data2(correl_dataset_dict, embs_dict, incl_header=True, score_type='similarity', data_has_header=False, score_index=2)
    # correlation_data2(correl_dataset_dict, save_file, vocab_path, embs_dict, incl_header=True, score_type='similarity', data_has_header=False, score_index=2)
    # calculate_missing_values('correl_ws_rel')
    correl_coeffs = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    print('Correl coeffs:', correl_coeffs)
    '''
    
    '''
    Calculate word pair distances
    
    include_base = True
    word_pair_file_dict = get_word_pair_datasets()
    embs_dict = get_model_path('embeds', ratio_models=ratio_models, include_base=include_base)
    vocab_file = get_vocab_num(7)
    save_file = get_syn_dists_basedir() + 'ratios_syn_dists_base.csv'
    
    print('word_pair_dict', word_pair_file_dict)
    print('embds', embs_dict)
    
    # word_pair_dists_dict(word_pair_file_dict['synonyms'], vocab_file, embs_dict, save_file, focus_col='focus_word', target_col='synonym')
    
    # PLOT WORD PAIR DISTANCES
    dist_file_dict = get_syn_dists_basedir() + 'all_syn_dists.csv',
    # save_fig_file = 'plots/word_pair_dists/w2vi_syn_dists_hist.png'
    # plot_dists_histogram_csv(dist_file, save_file=save_fig_file, plot=False, title='W2V init - Synonym pairs dists. histogram', model_filter='w2vi', dist_type='_cos')
    
    vocab_num = 7
    # correl_coeffs = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    # print('correl_coeffs', correl_coeffs)
    '''
    
    
    '''
    PLOT DISTANCE MEAN AND STD
    models = 'all' # 'all'
    dist_file_dict = {  'syn'  : get_syn_dists_basedir() + models + '_syn_dists.csv',
                        'nat'  : get_syn_dists_basedir() + models + '_nat_dists.csv',
                        'rand' : get_syn_dists_basedir() + models + '_rand_dists.csv'}
    model_filter = '_euc'
    mean_var_dict = word_pair_dists_mean_var(dist_file_dict, model_filter=model_filter)
    save_fig_file = 'plots/word_pair_dists/mean_std_' + models + '_dists_euc.png'
    
    print('mean_var_dict', mean_var_dict)
    
    plot_mean_vars(mean_var_dict, save_fig_file, model_filter=model_filter, plot=False)
    '''
    
    '''
    # MATRIX NORMS
    ratio_models = False
    sing_vals_dict = get_model_path('sing_vals', ratio_models=ratio_models)
    print('sing_vals_dict', sing_vals_dict)
    
    for name, sing_val_file in sing_vals_dict.items():
        sing_vals = np.load(sing_val_file)
        print(name, sing_vals[0])
    '''
    
    elapsed_time = time.time() - start_time
    print('\nTotal elapsed time: ', elapsed_time)
    
    # plt.show()