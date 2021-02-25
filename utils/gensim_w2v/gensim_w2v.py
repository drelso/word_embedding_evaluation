###
#
# Get Word2Vec vectors
#
###

import gensim.downloader as api
import csv
import numpy as np

def get_word_vector(word, model):
    try:
        model[word]
    except:
        print('Word not found')
        return False
    return model[word]
    
    
if __name__ == '__main__':
    #api.info()  # return dict with info about available models/datasets
    #api.info("text8")  # return dict with info about "text8" dataset

    model = api.load("word2vec-google-news-300")
    
    missing_words = 0
    
    with open('../data/dataset/vocabulary.csv','r') as v, \
        open('word_vectors.csv', 'w+') as f:
        
        word_vec_dim = 300
        
        vocab_data = csv.reader(v)
        wr = csv.writer(f)#, quoting=csv.QUOTE_ALL)
        
        # i = 0
        
        word_vec_arr = []
        
        for row in vocab_data:
            word_vec = get_word_vector(row[0], model)
            
            if isinstance(word_vec, bool):
                missing_words += 1
                print(row[0])
                word_vec = [0] * word_vec_dim
            
            temp = [row[0]] + [d for d in word_vec]
            word_vec_arr.append(temp)
            
            # i += 1
            # if i > 100: break
        
        wr.writerows(word_vec_arr)
            
        
    print('Missing %d words' % (missing_words))
    
    
    '''
    ### GET MISSING WORDS AND WRITE THEM IN A SEPARATE FILE
    with open('word2vec-google-news-300_voc3.csv', 'r') as f, \
    open('word2vec-google-news-300_voc3_MISS.csv', 'w+') as m:
    data = csv.reader(f)
    wr = csv.writer(m)
    missing_words = []
    for row in data:
        flag = False
        for i in range(1, len(row)):
            if float(row[i]) != 0:
                flag = True
                break
        if not flag:
            missing_words.append([row[0]])
    
    print(missing_words[:10])
    wr.writerows(missing_words)
    '''