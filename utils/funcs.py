### Auxiliary embedding functions

import csv
from collections import Counter

import torchtext
import torch


def build_vocabulary(counts_file, vocab_ixs_file, min_freq=1):
    ''''
    Builds a torchtext.vocab object from a CSV file of word
    counts and an optionally specified frequency threshold

    Requirements
    ------------
    import csv
    from collections import Counter
    import torchtext
    
    Parameters
    ----------
    counts_file : str
        path to counts CSV file
    min_freq : int, optional
        frequency threshold, words with counts lower
        than this will not be included in the vocabulary
        (default: 1)
    
    Returns
    -------
    torchtext.vocab.Vocab
        torchtext Vocab object
    '''
    counts_dict = {}

    print(f'{"@" * 30}\nVOCABULARY CONSTRUCTION\n{"@" * 30}')
    print(f'Constructing vocabulary from counts file in {counts_file}')

    num_inc = 0
    num_exc = 0

    with open(counts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # FIRST COLUMN IS ASSUMED TO BE THE WORD AND
            # THE SECOND COLUMN IS ASSUMED TO BE THE COUNT
            w_count = int(row[1])
            counts_dict[row[0]] = w_count
            if w_count < min_freq:
                num_exc += w_count
            else:
                num_inc += w_count

    counts = Counter(counts_dict)
    del counts_dict
    
    vocabulary = torchtext.vocab.Vocab(counts, min_freq=min_freq, specials=['<unk>', '<sos>', '<eos>', '<pad>'])
    perc_toks = "{:.2f}".format((len(vocabulary) / len(counts)) * 100) + '%'
    print(f'{len(vocabulary)} unique tokens in vocabulary with minimum frequency {min_freq} ({perc_toks} of {len(counts)} unique tokens in full dataset)')

    perc_inc = "{:.2f}".format((num_inc / (num_inc + num_exc)) * 100) + '%'
    print(f'{num_inc} of {(num_inc + num_exc)} words, vocabulary coverage of {perc_inc}')
    
    # SAVE LIST OF VOCABULARY ITEMS AND INDICES TO FILE
    with open(vocab_ixs_file, 'w+', encoding='utf-8') as v:
        vocabulary_indices = [[i, w] for i,w in enumerate(vocabulary.itos)]
        print(f'Writing vocabulary indices to {vocab_ixs_file}')
        csv.writer(v).writerows(vocabulary_indices)

    return vocabulary


def get_word2vec_vectors(gensim_path='data/word_embeddings/word2vec-google-news-300.csv'):
    """
    Load a CSV of word2vec Gensim vectors
    
    Requirements
    ------------
    import csv
    import torch
    
    Parameters
    ----------
    gensim_path : str
        filepath to the CSV embeddings file
    
    Returns
    -------
    torch.tensor
        tensor of word embeddings, dimensions
        are num_words x 300
    """
    with open(gensim_path, 'r') as v:
        i = 0
        word_vectors = []
        vocab = csv.reader(v)
        for row in vocab:
            vec = [float(d) for d in row[1:]]
            word_vectors.append(vec)
    
    return torch.tensor(word_vectors)


def print_parameters(parameters):
    '''
    Pretty print all model parameters

    Parameters
    ----------
    parameters : {str : X }
        parameter dictionary, where the keys are
        the parameter names with their corresponding
        values
    '''
    
    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        # num_tabs = int((32 - len(name))/8) + 1
        # tabs = '\t' * num_tabs
        num_spaces = 30 - len(name)
        spaces = ' ' * num_spaces
        print(f'{name}: {spaces} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')