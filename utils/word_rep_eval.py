###
#
# Word Representation Evaluation Functions
#
###

import os

from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import scipy.spatial


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['serif']
rcParams['font.serif'] = ['Times New Roman']



def set_axis_style(ax, labels, xlabel):
    ax.xaxis.set_tick_params(direction='out', labelrotation=90)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(xlabel)


def plot_word_pair_dists(word_pair_dists_file, save_file, fig_title='Word Pair Distance Distributions', dist_type='cos', colors=['#CF5C36', '#050517', '#05A8AA'], plot=False):
    if not os.path.exists(save_file + '.png'):
        print(f'No word pair {dist_type} distances plot file found at {save_file}, plotting word pair distances.')

        print(f"Loading word pair distances from {word_pair_dists_file}")

        word_pair_dists = np.load(word_pair_dists_file, allow_pickle=True)
        print(f"Plotting word pair {dist_type} distances for {len(word_pair_dists.item())} models: {word_pair_dists.item().keys()}")

        fig, axes = plt.subplots(nrows=1, ncols=len(word_pair_dists.item()), figsize=(9, 4), sharey=True)

        fig.suptitle(fig_title)

        for model_num, (model_name, dist_data) in enumerate(word_pair_dists.item().items()):
            print(f"Processing distances for {model_name} model")

            word_pair_types = ['random', 'context', 'synonym']
            
            dists = [[x for x in dist_data[word_pair_type][dist_type] if not np.isnan(x)] for word_pair_type in word_pair_types]

            for type_num, word_pair_type in enumerate(word_pair_types):
                print(f"\t- Processing {len(dist_data[word_pair_type][dist_type])} distances")
            
            if model_num == 0:
                axes[model_num].set_ylabel('Word Pair Distance')
            else:
                axes[model_num].spines["left"].set_visible(False)
                axes[model_num].yaxis.set_visible(False)
            
            axes[model_num].spines["right"].set_visible(False)
            axes[model_num].spines["top"].set_visible(False)
            axes[model_num].tick_params(axis='x', length=0)

            parts = axes[model_num].violinplot(dists, showmeans=True, showextrema=True)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor(colors[i])
                pc.set_alpha(0.7)

            parts['cbars'].set_visible(False)
            parts['cmeans'].set_visible(False)
            parts['cmaxes'].set_visible(False)
            parts['cmins'].set_visible(False)

            quartile_25s = []
            medians = []
            quartile_75s = []

            for data_part in dists:
                quartile_25, median, quartile_75 = np.percentile(data_part, [25, 50, 75], axis=0)
                quartile_25s.append(quartile_25)
                medians.append(median)
                quartile_75s.append(quartile_75)

            inds = np.arange(1, len(medians) + 1)
            axes[model_num].scatter(inds, medians, marker='o', facecolors='white', edgecolors='black', s=50, zorder=3)
            # ax.vlines(inds, quartile_25s, quartile_75s, color='k', linestyle='-', lw=5)

            set_axis_style(axes[model_num], word_pair_types, model_name)
            
            # Adjust plot to fit labels
            plt.tight_layout()
            
            print(f"Saving plot to {save_file}")
            plt.savefig(save_file)
        
            if plot: plt.show()
    else:
        print(f'Word pair {dist_type} distances plot found at {save_file}.')



def word_pair_distances(word_pair_file, embed_files_dict, save_file):
    if not os.path.exists(save_file):
        print(f'No word pair distances file found at {save_file}, creating an word pair distances file.')

        # Load NumPy dictionary of word pairs and convert it into
        # a Pandas DataFrame
        word_pairs_npy = np.load(word_pair_file, allow_pickle=True)
        
        word_pair_dists = {}
        word_pair_types = ['random', 'context', 'synonym']

        for embed_name, embed_file in embed_files_dict.items():
            print(f"Loading {embed_name} word representations")
            embeds = np.load(embed_file, allow_pickle=True)

            word_pair_dists[embed_name] = { word_pair_type: { 'euc': [], 'cos': [] } for word_pair_type in word_pair_types}

            # Change orientation so that every word pair is a row and columns are:
            # |  base_word  |  random  |  context  |  synonym  |
            # Only load the rows that have no null values
            word_pairs_df = pd.DataFrame.from_dict(word_pairs_npy.item(), orient='index').dropna()
            print(f"Loaded word pair dataframe for {embed_name} representations with {word_pairs_df.shape[0]} rows")

            missing_words = 0
            zero_embeds = 0
            
            for base_word, word_pairs in word_pairs_df.iterrows():
                # Only calculate the distances if all words are
                # in the vocabulary
                if base_word in embeds.item().keys() and \
                    word_pairs['random'] in embeds.item().keys() and \
                    word_pairs['context'] in embeds.item().keys() and \
                    word_pairs['synonym'] in embeds.item().keys():

                    base_embed = embeds.item()[base_word]

                    # Check if base embedding is not all zeros
                    if np.any(base_embed):

                        for word_pair_type in word_pair_types:
                            pair_embed = embeds.item()[word_pairs[word_pair_type]]
                            
                            # Check if pair embedding is not all zeros
                            if np.any(pair_embed):
                                euc_dist = scipy.spatial.distance.euclidean(base_embed, pair_embed)
                                cos_dist = scipy.spatial.distance.cosine(base_embed, pair_embed)

                                if not np.isnan(euc_dist):
                                    word_pair_dists[embed_name][word_pair_type]['euc'].append(euc_dist)
                                else:
                                    print(f"Euclidean distance is NaN for embeddings of words {base_word} and {word_pairs[word_pair_type]}: \n\n{base_embed[:10]} \n\n{pair_embed[:10]}")

                                if not np.isnan(cos_dist):
                                    word_pair_dists[embed_name][word_pair_type]['cos'].append(cos_dist)
                                else:
                                    print(f"Cosine distance is NaN for embeddings of words {base_word} and {word_pairs[word_pair_type]}: \n\n{base_embed[:10]} \n\n{pair_embed[:10]}")
                            else:
                                # print(f"{word_pairs[word_pair_type]} has zero embedding")
                                # If base embedding is zero, increment by number of
                                # word pair types
                                zero_embeds += len(word_pair_types)
                    else:
                        # print(f"{base_word} has zero embedding")
                        zero_embeds += 1
                else:
                    missing_words += 1
        
            print(f"\nFinished calculating {embed_name} distances, skipped {missing_words} word pairs and {zero_embeds} zero vector embeddings\n")

        print(f"Saving word pair distances to {save_file}")
        np.save(save_file, word_pair_dists)
    else:
        print(f'Word pair distances file found at {save_file}.')


def get_word_pairs(num_pairs, vocabulary, skipgram_data_path, save_file):
    if not os.path.exists(save_file):
        print(f'No word pair file found at {save_file}, creating an word pair file.')
        
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
            # Only include immediate context words
            if word in remaining_context_words and abs(ctx_pos) == 1:
                # print(focus, ctx, ctx_pos)
                # Only include a context word if it is in the vocabulary
                if ctx_word[0] in vocabulary.stoi.keys():
                    word_pairs[word]['context'] = ctx_word[0]
                    remaining_context_words.pop(remaining_context_words.index(word))
        
        print(f"{len(remaining_context_words)} remaining context words: {remaining_context_words}")

        print("Processing synonym pairs")

        for word in word_pairs.keys():
            synonym = get_synonym(word, vocabulary, syn_selection='s1')

            if synonym and (synonym != word):
                word_pairs[word]['synonym'] = synonym
                remaining_synonym_words.pop(remaining_synonym_words.index(word))

        print(f"{len(remaining_synonym_words)} remaining synonym words: {remaining_synonym_words}")
        
        print(f"Done processing word pairs, saving {len(word_pairs)} word pairs to {save_file}")

        np.save(save_file, word_pairs)
        # return word_pairs
    
    else:
        print(f'Word pair file found at {save_file}.')


def get_synonym(word, vocabulary, syn_selection='s1'):
    synsets = wn.synsets(word)
    # Single occurrence of every synonym
    synonym_set = set()
    
    for syn in synsets:
        for lem in syn.lemmas():
            # Don't add synonym if it is the same word
            if lem.name() != word:
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


import csv

def voc_and_array_to_embs_dict(vocab_file, npy_embs_file, save_file):
    npy_embs = np.load(npy_embs_file)
    voc = []
    with open(vocab_file,'r') as v:
        voc_reader = csv.reader(v)
        # Every row in this vocabulary is a tuple of the form:
        #   ['index', 'word']
        voc = [v for v in voc_reader]
    
    embs_dict = {}
    for i, v in voc:
        embs_dict[voc] = npy_embs[int(i)]
    
    np.save(save_file, embs_dict)