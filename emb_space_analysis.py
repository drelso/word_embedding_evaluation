###
#
# Embedding space analysis
#
###

import time
import numpy as np
import torch

import csv
import re

from nltk.corpus import wordnet as wn

# from sklearn.decomposition import TruncatedSVD
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from utils.funcs import npy_to_tsv
from utils.model_paths import get_model_path, get_vocab_for_name, get_vocab_num, get_correl_dict, get_base_embeds, get_word_pair_datasets, get_syn_dists_basedir, get_base_embeds#, which_files_exist
# from utils.spatial import pairwise_dists, explained_variance, get_embedding_norms, calculate_singular_vals
from utils.plot import plot_exp_variance, plot_dists_dict_histogram, plot_histogram, plot_exp_variances, plot_dists_dict_histo_subplot, plot_dists_histogram_csv, word_pair_dists_mean_var, plot_mean_vars
from utils.data_processing import calculate_histograms, calculate_missing_values
from utils.validation import word_pair_distances, word_pair_dists_dict, correl_coefficients_for_vocab


from utils.funcs import print_parameters
from utils.word_rep_funcs import build_vocabulary
from config import parameters









def get_syn_pairs(data_file, save_file, randomly_sampled=0, max_datapoints=100000, target_col='synonym'):
    """
    Get a set of synonym pairs together with
    their counts and frequencies. The data file
    is expected in the following format:
    - synonym
    - context_word
    - sent_num
    - focus_index
    - context_position
    - focus_word
    - book_number
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    data_file : str
        path to the data file
    save_file : str
        path to the file to save the synonyms to
    randomly_sampled : int, optional
        if non-zero, number of samples to randomly
        select from the dataset (default: 0)
    max_datapoints : int, optional
        if randomly sampling datapoints, this
        value defines the range to sample from,
        if this is lower than the number of
        datapoints the last points will not be
        sampled, if it is larger than the number
        of datapoints the returned samples will
        be fewer than the value specified in
        randomly_sampled (default: 100000)
    target_col : str, optional
        name of the target word column
        (default: 'synonym')
    """
    
    keep_samples = np.array([])
    max_index = 0
    
    if int(randomly_sampled) > 0:
        keep_samples = np.random.choice(max_datapoints, randomly_sampled, replace=False)
        max_index = max(keep_samples)
        print('Keep samples: ', keep_samples)
        
    with open(data_file, 'r') as f, \
        open(save_file, 'w+') as s:
        data = csv.reader(f)
        writer = csv.writer(s)
        
        header = next(data)
        cols = {w:i for i,w in enumerate(header)}
        
        # i = 0
        
        syns = []
        syn_counts = []
        count_index = 1
        
        # syn_counts.append(['focus_word', 'synonym', 'counts'])
        
        for i, row in enumerate(data):
            if not keep_samples.any() or i in keep_samples:
                syn_pair = [row[cols['focus_word']], row[cols[target_col]]]
                if syn_pair in syns:
                    # Add 1 to account for the header row
                    # syn_index = syns.index(syn_pair) + 1
                    syn_index = syns.index(syn_pair)
                    count = syn_counts[syn_index][2]
                    syn_counts[syn_index][2] = int(count) + 1
                else:
                    syns.append(syn_pair)
                    syn_counts.append([syn_pair[0], syn_pair[1], 1])
                # i+=1
                # if i > 20: break
            if i % 100000 == 0:
                print('%d word pairs processed' % (i))
            
            if max_index > 0 and i > max_index: break
            
        syn_freqs = [[s[0], s[1], s[2], s[2]/i] for s in syn_counts]
        
        print('Processed %d word pairs (%d unique)' % (i, len(syn_freqs)))
        print('Saving file to ', save_file)
        
        writer.writerow(['focus_word', target_col, 'counts', 'freqs'])
        writer.writerows(syn_freqs)


def word_pair_distances(word_pair_file, vocab_file, embs_list, save_file, focus_col='focus_word', target_col='synonym'):
    """
    Given a file of word pairs, calculate the
    distances between their different embeddings
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    word_pair_file : str
        path to the file containing the
        (unique) word pairs. This file should
        have (at least) the following named
        columns:
        - focus_col
        - target_col
    vocab_file : str
        path to the vocabulary file
    embs_list : list[[str,str]]
        list of tuples where the first element
        is the name of the embeddings and the
        second is the path to the word embeddings
        (requires NPY format)
    save_file : str
        path to the file to save the calculated
        distances to
    focus_col : str, optional
        name of the column containing the
        'focus' word (default: 'focus_word')
    target_col : str, optional
        name of the column containing the
        'target' word (default: 'synonym')
    
    NOTE: if there is a zero vector we ignore
    the word-synonym distance row. This can be
    changed later
    """
    with open(word_pair_file, 'r') as f, \
        open(vocab_file, 'r') as v, \
        open(save_file, 'w+') as s:
        
        data = csv.reader(f)
        header = next(data)
        vocabulary = {w[0]: i for i,w in enumerate(csv.reader(v))}
        writer = csv.writer(s)
        
        try:
            focus_index = header.index(focus_col)
        except:
            raise ValueError('Focus column %r not in header: %r' % (focus_col, header)) from None
        
        try:
            target_index = header.index(target_col)
        except:
            raise ValueError('Target column %r not in header: %r' % (target_col, header)) from None
        
        embs = {name: np.load(path) for name, path in embs_list}
        emb_names = list(embs.keys())
        cols = [focus_col, target_col]
        
        for name in emb_names:
            cols.append(name + '_cos')
            cols.append(name + '_euc')
        
        print(cols)
        dist_matrix = [cols]
        
        missing_focus = 0
        missing_targets = 0
        zero_vecs = 0
        
        i = 0
        
        for row in data:
            try: focus = vocabulary[row[focus_index]]
            except:
                missing_focus += 1
                continue
            try: target = vocabulary[row[target_index]]
            except:
                missing_targets += 1
                continue
            
            dists = [row[focus_index], row[target_index]]
            is_zero = False
            
            for name in embs.keys():
                emb_1 = embs[name][focus]
                emb_2 = embs[name][target]
                cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)        
                
                dists.append(cos_dist)
                dists.append(euc_dist)
            
                if np.isnan(cos_dist):
                    zero_vecs += 1
                    is_zero = True
            
            if not is_zero: dist_matrix.append(dists)
            
            i += 1
        
        print('Total words: %d \t Missing words: focus=%d \t synonyms=%d \t zero vecs=%d' % (i, missing_focus, missing_targets, zero_vecs))
        
        print('Saving distances to %r' % (save_file))
        writer.writerows(dist_matrix)


def word_pair_dists_dict(word_pair_file, vocab_file, embs_dict, save_file, focus_col='focus_word', target_col='synonym'):
    """
    Given a file of word pairs, calculate the
    distances between their different embeddings
    
    Requirements
    ------------
    import csv
    import numpy as np
    import scipy.spatial
    
    Parameters
    ----------
    word_pair_file : str
        path to the file containing the
        (unique) word pairs. This file should
        have (at least) the following named
        columns:
        - focus_col
        - target_col
    vocab_file : str
        path to the vocabulary file
    embs_list : dict{str: str}
        dictionary of embedding paths where the
        key is the name of the embeddings and the
        value is the path to the word embeddings
        (requires NPY format)
    save_file : str
        path to the file to save the calculated
        distances to
    focus_col : str, optional
        name of the column containing the
        'focus' word (default: 'focus_word')
    target_col : str, optional
        name of the column containing the
        'target' word (default: 'synonym')
    
    NOTE: if there is a zero vector we ignore
    the word-synonym distance row. This can be
    changed later
    """
    with open(word_pair_file, 'r') as f, \
        open(vocab_file, 'r') as v, \
        open(save_file, 'w+') as s:
        
        data = csv.reader(f)
        header = next(data)
        vocabulary = {w[0]: i for i,w in enumerate(csv.reader(v))}
        writer = csv.writer(s)
        
        try:
            focus_index = header.index(focus_col)
        except:
            raise ValueError('Focus column %r not in header: %r' % (focus_col, header)) from None
        
        try:
            target_index = header.index(target_col)
        except:
            raise ValueError('Target column %r not in header: %r' % (target_col, header)) from None
        
        embs = {name: np.load(path) for name, path in embs_dict.items()}
        emb_names = list(embs.keys())
        cols = [focus_col, target_col]
        
        for name in emb_names:
            cols.append(name + '_cos')
            cols.append(name + '_euc')
        
        print(cols)
        dist_matrix = [cols]
        
        missing_focus = 0
        missing_targets = 0
        zero_vecs = 0
        
        i = 0
        
        for row in data:
            try: focus = vocabulary[row[focus_index]]
            except:
                missing_focus += 1
                continue
            try: target = vocabulary[row[target_index]]
            except:
                missing_targets += 1
                continue
            
            dists = [row[focus_index], row[target_index]]
            is_zero = False
            
            for name in embs.keys():
                emb_1 = embs[name][focus]
                emb_2 = embs[name][target]
                cos_dist = scipy.spatial.distance.cosine(emb_1, emb_2)
                euc_dist = scipy.spatial.distance.euclidean(emb_1, emb_2)        
                
                dists.append(cos_dist)
                dists.append(euc_dist)
            
                if np.isnan(cos_dist):
                    zero_vecs += 1
                    is_zero = True
            
            if not is_zero: dist_matrix.append(dists)
            
            i += 1
        
        print('Total words: %d \t Missing words: focus=%d \t synonyms=%d \t zero vecs=%d' % (i, missing_focus, missing_targets, zero_vecs))
        
        print('Saving distances to %r' % (save_file))
        writer.writerows(dist_matrix)
        
    
def calc_dist_changes(dist_file, emb_source, emb_target, title='', focus_col='focus_word', target_col='synonym'):
    """
    Given a file of distances, calculate
    the changes in distance between source
    and target embeddings in the following
    way:
    
        dist_change = source_dist - target_dist
    
    Such that a positive value implies that
    the source distance is larger than the
    target distance.
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    dist_file : str
        path to the file containing the
        distances. The file must be a CSV
        file with, at least, the following
        columns:
        - focus_col
        - target_col
        - @emb_source
        - @emb_target
    emb_source : str
        name of the source embedding distances
        column in the distances file
    emb_target : str
        name of the target embedding distances
        column in the distances file
    title : str, optional
        title to print to console
    focus_col : str, optional
        name of focus word column
        (default: 'focus_word')
    target_col : str, optional
        name of target word column
        (default: 'synonym')
    
    Returns
    -------
    list[float]
        list of distances sorted in ascending
        order
    """
    with open(dist_file, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        
        # Creates a dictionary of column names
        cols = {w: i for i, w in enumerate(header)}
        
        dists = []
        dist_changes = []
        
        for row in data:
            word_pair = row[cols[focus_col]] + '-' + row[cols[target_col]]
            dist_change = float(row[cols[emb_source]]) - float(row[cols[emb_target]])
            
            dists.append([word_pair, dist_change])
            dist_changes.append(dist_change)
            
        sorted_dists = sorted(dists, key=lambda x: x[1])
        
        total_change = np.sum(dist_changes)
        average_change = total_change / len(dist_changes)
        
        print('Distance changes ', title)
        print('Total change: ', total_change)
        print('Average change: ', average_change)
        
        return sorted_dists


def rand_word_pairs(vocab_path, save_file, num_word_pairs=100, incl_header=True, focus_col='focus_word', target_col='rand_word'):
    """
    Construct a set of randomly generated
    word pairs from a vocabulary
    
    Requirements
    ------------
    import csv
    import numpy as np
    
    Parameters
    ----------
    vocab_path : str
        path to the vocabulary file, which
        is assumed to be a CSV file with a
        header where the first column corresponds
        to the words
    save_file : str
        file to save the random word pairs to
    num_word_pairs : int, optional
        number of word pairs to generate
        (default: 100)
    """
    
    with open(vocab_path, 'r') as v, \
        open(save_file, 'w+') as s:
        
        voc_data = csv.reader(v)
        writer = csv.writer(s)
        
        if incl_header: header = next(voc_data)
        vocab = {i: w[0] for i, w in enumerate(voc_data)}
        
        # num_word_pairs x 2 matrix of random indeces
        rand_ixs = np.random.choice(len(vocab), [num_word_pairs, 2])
        
        word_pairs = [[focus_col, target_col]]
        
        for w1, w2 in rand_ixs:
            word_pairs.append([vocab[w1],vocab[w2]])
        
        print('Saving %d randomly generated word pairs to %r' % (len(word_pairs), save_file))
        writer.writerows(word_pairs)




def get_word_pairs(num_pairs, vocabulary, skipgram_data_path, save_file):
    vocab_size = len(vocabulary)
    
    rand_word_ixs = np.random.choice(vocab_size, num_pairs, replace=False) # p=None)
    rand_word_pair_ixs = np.random.choice(vocab_size, num_pairs, replace=False) # p=None)
    
    remaining_context_words = [vocabulary.itos[ix] for ix in rand_word_ixs]
    remaining_synonym_words = [vocabulary.itos[ix] for ix in rand_word_ixs]
    
    print("Initialising word pair dictionary and calculating random word pairs")

    # Initialise dictionary
    word_pairs = {vocabulary.itos[ix]: {'random': vocabulary.itos[rand_word_pair_ixs[i]], 'context': None, 'synonym': None} for i, ix in enumerate(rand_word_ixs)}

    # Get context words
    skipgram_dataset = np.load(skipgram_data_path, allow_pickle=True)

    print(f"Processing contextual pairs from Skip-gram dataset at {skipgram_data_path}")

    for focus, ctx_word, sent_num, focus_ix, ctx_pos in skipgram_dataset:
        word = focus[0]
        if word in remaining_context_words and abs(ctx_pos) == 1:
            # print(focus, ctx, ctx_pos)
            word_pairs[word]['context'] = ctx_word[0]
            remaining_context_words.pop(remaining_context_words.index(word))
    
    print(f"{len(remaining_context_words)} remaining context words: {remaining_context_words}")

    print("Processing synonym pairs")

    for word in word_pairs.keys():
        synonym = get_synonym(word, vocabulary, syn_selection='s1')

        if synonym:
            word_pairs[word]['synonym'] = synonym
            remaining_synonym_words.pop(remaining_synonym_words.index(word))

    print(f"{len(remaining_synonym_words)} remaining synonym words: {remaining_synonym_words}")
    
    print(f"Done processing word pairs, saving {len(word_pairs)} word pairs to {save_file}")

    np.save(save_file, word_pairs)
    return word_pairs


def get_synonym(word, vocabulary, syn_selection='s1'):
    synsets = wn.synsets(word)
    # Single occurrence of every synonym
    synonym_set = set()
    
    for syn in synsets:
        for lem in syn.lemmas():
            # Don't add synonym if it is the same word
            if lem == word: continue
            # Get the synonym in lowercase
            synonym_set.add(lem.name().lower())

    synonym = synonym_selection(
                synonym_set,
                vocabulary,
                syn_selection=syn_selection)

    return synonym
    


def synonym_selection(synonym_set, vocabulary, syn_selection='s1'):
    """
    Given a set of synonyms, select one
    or a subset based on specific criteria
    like their frequency in the data or
    random selection
    
    Requirements
    ------------
    import numpy as np
    import random
    import csv
    
    Parameters
    ----------
    synonym_set : set
        set of available synonyms
    vocabulary : torchtext.Vocab
        torchtext vocabulary object
    syn_selection : str, optional
        synonym selection strategy, possible
        values are:
        - ml - maximum likelihood
        - s1 - randomly sample one
        - sn - randomly sample any number of syns
        - sw - randomly sample one (weighted by freq)
        - swn - randomly sample any number of syns
                (weighted by freq)
        (default: 's1')
    """
    potential_syns = []
    syn_freqs = []
    
    for syn in synonym_set:
        # if syn in vocabulary:
        if syn in vocabulary.stoi.keys():
            potential_syns.append(syn)
            syn_freqs.append(vocabulary.freqs[syn])

    syn_freqs /= np.sum(syn_freqs)

    if len(potential_syns) == 0: return []
    if len(potential_syns) == 1: return potential_syns[0]

    # Maximum likelihood, select synonym with
    # the highest frequency in the vocabulary
    if syn_selection == 'ml':
        return potential_syns[np.argmax(syn_freqs)]

    # Pick one synonym randomly
    if syn_selection == 's1':
        return np.random.choice(potential_syns)
    
    # Pick one synonym randomly, weighted by
    # its frequency in the dataset
    if syn_selection == 'sw':
        return np.random.choice(potential_syns, p=syn_freqs)
    
    num_syns = np.random.randint(1, len(potential_syns)+1)
    
    # Pick a random number of synonyms randomly
    if syn_selection == 'sn':
        return np.random.choice(potential_syns, size=num_syns, replace=False)
    
    # Pick a random number of synonyms randomly,
    # weighted by their frequencies in the dataset
    if syn_selection == 'swn':
        return np.random.choice(potential_syns, p=syn_freqs, size=num_syns, replace=False)
    else:
        raise ValueError(f'Unrecognised syn_selection {syn_selection}')




if __name__ == '__main__':
    
    start_time = time.time()
    
    ratio_models = False
    
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