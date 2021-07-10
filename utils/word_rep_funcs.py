###
#
# Baseline Word Representations
# Auxiliary functions
#
###

import os
import re
import csv
from collections import Counter, OrderedDict

import numpy as np

import gensim.downloader as api
import torchtext
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn # process_data
from nltk.corpus import framenet as fn

def build_vocabulary(counts_file, min_freq=1):
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

    print(f'Constructing vocabulary from counts file in {counts_file}')

    with open(counts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # FIRST COLUMN IS ASSUMED TO BE THE WORD AND
            # THE SECOND COLUMN IS ASSUMED TO BE THE COUNT
            counts_dict[row[0]] = int(row[1])

    counts = Counter(counts_dict)
    del counts_dict
    
    vocabulary = torchtext.vocab.Vocab(counts, min_freq=min_freq, specials=['<unk>', '<sos>', '<eos>', '<pad>'])
    print(f'{len(vocabulary)} unique tokens in vocabulary with (with minimum frequency {min_freq})')
    
    return vocabulary



## Word2Vec embeddings
def word2vec_with_vocab(vocabulary, save_file, embedding_dim=300):
    '''
    Extracts Word2Vec embeddings for a given
    vocabulary and saves them into a NPY file
    as a dictionary where the word string is the
    key and the word vector as an array of floats
    is the value. This takes the form:
    {
        'word' : [float],
        ...
    }

    NOTE: words that do not appear in the Word2Vec
          vocabulary are mapped to the zero vector.
          A list of all missing words is stored in
          the same location as the embeddings file
          with the same file name with '_missing_words'
          as a suffix

    Requirements
    ------------
    import os
    import numpy as np

    Parameters
    ----------
    vocabulary : torchtext.vocab
        vocabulary object to convert to Word2Vec
        embeddings
    save_file : str
        path to save the embeddings to
    '''
    if not os.path.exists(save_file):
        print(f'No Word2Vec embeddings file found at {save_file}, creating an embeddings file.')
        model = api.load("word2vec-google-news-300")
    
        embeddings = {}
        missing_words = []
        

        for i in range(len(vocabulary.itos)):
            word = vocabulary.itos[i]
            try:
                word_vec = model[word]
                embeddings[word] = word_vec
            except:
                #print('Word not found')
                missing_words.append(word)
                embeddings[word] = [0.] * embedding_dim
        
        print(f"Saving {len(embeddings)} word embeddings to file at {save_file}")
        np.save(save_file, embeddings)

        print(f"({len(missing_words)} missing words set to zero vectors)")
        missing_words_file = save_file + '_missing_words'
        np.save(missing_words_file, missing_words)
    else:
        print(f'Word2Vec embeddings file found at {save_file}.')


## HellingerPCA embeddings
def hellingerPCA_with_vocab(vocabulary, source_vocab, source_vecs, save_file, embedding_dim=200):
    '''
    Extracts HellingerPCA (200 dims) embeddings for a
    given vocabulary and saves them into a NPY file
    as a dictionary where the word string is the
    key and the word vector as an array of floats
    is the value. This takes the form:
    {
        'word' : [float],
        ...
    }

    Embeddings are downloaded from http://www.lebret.ch/words/embeddings/200/

    NOTE: words that do not appear in the HellingerPCA
          vocabulary are mapped to the zero vector.
          A list of all missing words is stored in
          the same location as the embeddings file
          with the same file name with '_missing_words'
          as a suffix

    Requirements
    ------------
    import os
    import numpy as np

    Parameters
    ----------
    vocabulary : torchtext.vocab
        vocabulary object to convert to Word2Vec
        embeddings
    save_file : str
        path to save the embeddings to
    '''
    if not os.path.exists(save_file):
        print(f'No HellingerPCA embeddings file found at {save_file}, creating an embeddings file.')
        
        with open(source_vocab, 'r', encoding='utf-8') as v, \
            open(source_vecs, 'r') as vecs:
            s_vocab = v.read().splitlines() 
            s_vecs = vecs.read().splitlines()
        
        print(f'Num vocab words: {len(s_vocab)} {s_vocab[:10]}')
        print(f'Num word vecs: {len(s_vecs)} {s_vecs[0]}')
        
        embeddings = {}
        missing_words = []

        for i in range(len(vocabulary.itos)):
            word = vocabulary.itos[i]
            
            if word in s_vocab:
                word_ix = s_vocab.index(word)
                embeddings[word] = [float(x) for x in s_vecs[word_ix].split()]
            else:
                #print('Word not found')
                missing_words.append(word)
                embeddings[word] = [0.] * embedding_dim
        
        print(f"Saving {len(embeddings)} word embeddings to file at {save_file}")
        np.save(save_file, embeddings)

        print(f"({len(missing_words)} missing words set to zero vectors)")
        missing_words_file = save_file + '_missing_words'
        np.save(missing_words_file, missing_words)
    else:
        print(f'HellingerPCA embeddings file found at {save_file}.')


## GloVe embeddings
# def glove_with_vocab(vocabulary, source_vecs, save_file, embedding_dim=300):
#     '''
#     Extracts GloVe (300 dims) embeddings for a
#     given vocabulary and saves them into a NPY file
#     as a dictionary where the word string is the
#     key and the word vector as an array of floats
#     is the value. This takes the form:
#     {
#         'word' : [float],
#         ...
#     }

#     Embeddings are downloaded from https://nlp.stanford.edu/data/glove.840B.300d.zip

#     NOTE: words that do not appear in the GloVe
#           vocabulary are mapped to the zero vector.
#           A list of all missing words is stored in
#           the same location as the embeddings file
#           with the same file name with '_missing_words'
#           as a suffix

#     Requirements
#     ------------
#     import os
#     import numpy as np
#     import re

#     Parameters
#     ----------
#     vocabulary : torchtext.vocab
#         vocabulary object to convert to Word2Vec
#         embeddings
#     save_file : str
#         path to save the embeddings to
#     '''
#     if not os.path.exists(save_file):
#         print(f'No GloVe embeddings file found at {save_file}, creating an embeddings file.')
        
#         regex = re.compile('^[^\s]*\s')
#         s_vocab = {}
#         s_vecs = []
#         s_lines = []
        
#         with open(source_vecs, 'r', encoding='utf-8') as vecs_file:
#             # s_lines = vecs_file.read().splitlines()
#             for i, line in enumerate(vecs_file):
#                 if line:
#                     # line_list = line.split()
#                     # s_vecs.append(line_list)
#                     # word = line_list[0]
#                     s_lines.append(line)
#                     word = regex.match(line).group()[:-1]
#                     s_vocab[word] = i
                
#                 if not i % 100000: print(f'Processed {i} lines, current word {word}')
        
#         print('Loaded the file, lines read, processing...', flush=True)

#         # print(f'Num word vecs: {len(s_vecs)} of dimension {len(s_vecs[0]) - 1} \t {s_vecs[0]}')
#         print(f'Num words in GloVe vocabulary: {len(s_vocab)}', flush=True)
        
#         print(f'Index of "hello": {s_vocab["hello"]} \n {s_lines[s_vocab["hello"]]}')

#         # print(s_vocab)
#         embeddings = {}
#         missing_words = []

#         for i in range(len(vocabulary.itos)):
#             word = vocabulary.itos[i]
            
#             if word in s_vocab.keys():
#                 word_ix = s_vocab[word]
#                 word_list = s_lines[word_ix].split()
#                 print('\n\n', word_list)
#                 word_vec = word_list[1:]
#                 print('\n\n', word_vec)
#                 print('\n\n', word_list[0])
                
#                 # word_vec = s_vecs[word_ix]
#                 if not i % 2000: print(f'Processed {i} vocabulary items, current word {word}')
#                 if word != word_list[0]:
#                     raise ValueError(f'{word} does not match word in source: {word_list[0]}')
#                 embeddings[word] = [float(x) for x in word_vec]
#             else:
#                 #print('Word not found')
#                 missing_words.append(word)
#                 embeddings[word] = [0.] * embedding_dim
        
#         print(f"Saving {len(embeddings)} word embeddings to file at {save_file}")
#         np.save(save_file, embeddings)

#         print(f"({len(missing_words)} missing words set to zero vectors)")
#         missing_words_file = save_file + '_missing_words'
#         np.save(missing_words_file, missing_words)
#     else:
#         print(f'GloVe embeddings file found at {save_file}.')


## GloVe embeddings
def glove_with_vocab(vocabulary, save_file, embedding_dim=300):
    '''
    Extracts GloVe embeddings for a given
    vocabulary and saves them into a NPY file
    as a dictionary where the word string is the
    key and the word vector as an array of floats
    is the value. This takes the form:
    {
        'word' : [float],
        ...
    }

    NOTE: words that do not appear in the GloVe
          vocabulary are mapped to the zero vector.
          A list of all missing words is stored in
          the same location as the embeddings file
          with the same file name with '_missing_words'
          as a suffix

    Requirements
    ------------
    import os
    import numpy as np

    Parameters
    ----------
    vocabulary : torchtext.vocab
        vocabulary object to convert to GloVe
        embeddings
    save_file : str
        path to save the embeddings to
    '''
    if not os.path.exists(save_file):
        print(f'No GloVe embeddings file found at {save_file}, creating an embeddings file.')
        model = api.load("glove-wiki-gigaword-300")
    
        embeddings = {}
        missing_words = []
        
        for i in range(len(vocabulary.itos)):
            word = vocabulary.itos[i]
            try:
                word_vec = model[word]
                embeddings[word] = word_vec
            except:
                #print('Word not found')
                missing_words.append(word)
                embeddings[word] = [0.] * embedding_dim
        
        print(f"Saving {len(embeddings)} word embeddings to file at {save_file}")
        np.save(save_file, embeddings)

        print(f"({len(missing_words)} missing words set to zero vectors)")
        missing_words_file = save_file + '_missing_words'
        np.save(missing_words_file, missing_words)
    else:
        print(f'GloVe embeddings file found at {save_file}.')


## LIWC Vectors
def get_liwc_vectors(liwc_feature_file, prefix='liwc_'):
    print(f"Loading LIWC vectors from {liwc_feature_file}")
    with open(liwc_feature_file, 'r', encoding='utf-8') as l:
        # liwc_feats = l.read().splitlines()
        liwc_feats = csv.reader(l, delimiter='\t')

        processing_feats = True

        liwc_feats_dict = {}
        word_liwc_feats = {}

        for line in liwc_feats:
            if processing_feats:
                if len(line) == 2:
                    liwc_feats_dict[int(line[0])] = prefix + line[1]

            if len(line) == 1 and line[0] == '%':
                if len(liwc_feats_dict): processing_feats = False
            
            if not processing_feats:
                if len(line) >= 2:
                    word_liwc_feats[line[0]] = [liwc_feats_dict[int(f)] for f in line[1:]]
                
        # print(liwc_feats_dict)
        # print(word_liwc_feats)
    
    print(f"Processed LIWC vectors for {len(word_liwc_feats)} words with {len(liwc_feats_dict)} features.")

    return liwc_feats_dict, word_liwc_feats


def partial_match(word, partial_strings):
    initial = word[0]
    matches = []

    if initial in partial_strings.keys():
        for part in partial_strings[initial]:
            if word.beginswith(part):
                matches.append(part + '*')
    
    return matches



## Feature Vectors
def feature_vectors(vocabulary, liwc_file, word_feats_save_file):

    feature_dict = OrderedDict()

    # for feat in liwc_feats_dict:
    #     feature_dict[feat] = []

    # for word, feats in word_liwc_features.items():
    #     for feat in feats:
    #         if feat in feature_dict:
    #             feature_dict[feat].append(word)
    #         else:
    #             feature_dict[feat] = [word]
    
    # return True

    # MAKE A "COORDINATE" DICTIONARY: FOR EVERY FEATURE 
    # ADD THE INDEX OF THE WORD THAT APPLIES TO IT

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get LIWC feature dictionary, word features, and partial words
    liwc_feats_dict, word_liwc_features = get_liwc_vectors(liwc_file)

    # print(liwc_feats_dict)
    # print('\n\nword liwc features: \n\n')
    # print(word_liwc_features)

    partial_strings = {}

    for word in word_liwc_features.keys():
        if word.endswith('*'):
            # Get first letter of word and add
            # word to dictionary
            initial = word[0]
            if initial in partial_strings.keys():
                partial_strings[initial].append(word[:-1])
            else:
                partial_strings[initial] = [word[:-1]]

    # print('\n\npartial words: \n\n')
    # print(partial_strings)
    # exit()

    num_subwords = 0
    num_liwc_words = 0
    num_frames = 0
    num_wordnet = 0
    
    for i in range(len(vocabulary.itos)):
        word = vocabulary.itos[i]
        # toks = [tokenizer.ids_to_tokens[ix] for ix in tok_ids[1:-1]]
        # subword_set.update(toks)
        
        # Add subwords to feature dictionary
        tok_ids = tokenizer(word)['input_ids']
        sub_prefix = 'sub_'
        # Skip beginning [CLS] and end [SEP] tokens
        for tok_id in tok_ids[1:-1]:
            feat_name = sub_prefix + tokenizer.ids_to_tokens[tok_id]
            if feat_name in feature_dict:
                feature_dict[feat_name].append(i)
            else:
                feature_dict[feat_name] = [i]
                num_subwords += 1
        
        # Add word LIWC features to feature dictionary
        liwc_prefix = 'liwc_'
        # Start by matching partial strings, e.g. updat*
        if word[0] in partial_strings.keys():
            for part in partial_strings[word[0]]:
                if word.startswith(part):
                    num_liwc_words += 1
                    feat = liwc_prefix + part + '*'
                    if feat in feature_dict:
                        feature_dict[feat].append(i)
                    else:
                        feature_dict[feat] = [i]

        if word in word_liwc_features.keys():
            liwc_feats = word_liwc_features[word]
            num_liwc_words += 1
            for feat in liwc_feats:
                feat = liwc_prefix + feat
                if feat in feature_dict:
                    feature_dict[feat].append(i)
                else:
                    feature_dict[feat] = [i]

        # Add frames to feature dictionary
        # (currently only frame names)
        frames = fn.frames_by_lemma(r'(?i)'+ re.escape(word))
        frame_prefix = 'frame_'
        for frame in frames:
            feat = frame_prefix + frame.name
            if feat in feature_dict:
                feature_dict[feat].append(i)
            else:
                feature_dict[feat] = [i]
                num_frames += 1

        # Add synset knowledge to feature dictionary
        # (currently lexical names and POS tags)
        synsets = wn.synsets(word)
        synspos_prefix = 'syns-pos_'
        pos_set = set()
        synslex_prefix = 'syns-lex_'
        lex_set = set()
        synshyper_prefix = 'syns-hyper_'
        hyperset = set()

        for s in synsets:
            synset_els = s.lexname().split('.')
            if len(synset_els) != 2:
                print(f"Parsing error with synsets for {word}: {synset_els}")
            
            pos_set.add(synset_els[0])
            lex_set.add(synset_els[1])
            hyperset = set([h.name().split('.')[0] for h in synsets[0].hypernyms()])

        for pos in pos_set:
            feat = synspos_prefix + pos
            if feat in feature_dict:
                feature_dict[feat].append(i)
            else:
                feature_dict[feat] = [i]
                num_wordnet += 1
        
        for lex in lex_set:
            feat = synslex_prefix + lex
            if feat in feature_dict:
                feature_dict[feat].append(i)
            else:
                feature_dict[feat] = [i]
                num_wordnet += 1
            
        for hyper in hyperset:
            feat = synshyper_prefix + hyper
            if feat in feature_dict:
                feature_dict[feat].append(i)
            else:
                feature_dict[feat] = [i]
                num_wordnet += 1
        
        # TODO: WORDNET QUERIES:
        #       - POS TAGS
        #       - SUPERSENSES

        # TODO: OXFORD DICTIONARY API?

        # TODO: CHECK SEMCOR
    
    # print(feature_dict)

    print(f"Constructed feature dictionary with: \n\t{num_subwords} subwords \n\t{num_liwc_words} LIWC words \n\t{num_frames} frames \n\t{num_wordnet} WordNet features")

    print(f"Saving feature dictionary ({len(feature_dict)} features) to {word_feats_save_file}")
    np.save(word_feats_save_file, feature_dict)

    
def get_word_knowledge(word, verbose=False):
    """
    Query the FrameNet and WordNet
    information related to a given word
    
    Parameters
    ----------
    word : str
        word to query
    verbose : bool, optional
        whether to print the information
        obtained from the query (default: False)
    
    Returns
    -------
    object
        FrameNet object returned by query
    object
        WordNet synset object returned by
        query
    """
    frames = fn.frames_by_lemma(r'(?i)'+word)
    synsets = wn.synsets(word)
    
    if verbose:
        print('Frames: ', frames)
        print('Synsets: ', synsets)
    
    return frames, synsets