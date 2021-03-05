### Luke's Embedding Demo

import csv
import spacy
from sentence_transformers import SentenceTransformer

def read_luke_data(datafile):
    data = []
    with open(datafile, 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            if len(row) != 3:
                print(f'Row has length of {len(row)}, should be three. Skipping. \n\tWRONG ROW: {row}')
            
            # Removing first column, which is
            # some sort of index
            data.append([row[1], row[2]])
    
    return data

if __name__ == "__main__":
    # pip install spacy
    # python -m spacy download en_core_web_sm
    # pip install spacy-transformers
    # python -m spacy download en_core_web_trf

    datafile = 'data/Responses_sample.csv'

    data = read_luke_data(datafile)

    print(f'Read {len(data)} lines from dataset at {datafile}')
    
    ## Sentence Transformers
    ## from https://github.com/UKPLab/sentence-transformers
    ## and paper at https://arxiv.org/pdf/1908.10084.pdf
    ## pip install -U sentence-transformers

    # Full list of models is available in
    # https://www.sbert.net/docs/pretrained_models.html
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    sent_embs = []

    nlp = spacy.load('en_core_web_sm')

    for i, row in enumerate(data):
        # Every row in data has two columns:
        #   [label, sentence]
        # print(i, row[1])
        text = row[1]

        # Append tuple made up of sentence and embedding
        # (sentence_text, sentence_embedding_768_dims)
        # This embeds the full text, which is made up of
        # more than one sentence (always 2 sentences?)
        # sent_embs.append((text, model.encode(text)))

        # This part embeds the sentences separately
        doc = nlp(text)
        for j, sent in enumerate(doc.sents):
            print(j, sent)
            sent_embs.append((sent, model.encode(sent.text)))
        if i > 3: break
    
    print(f'sent_embs: {sent_embs[0][0]} embedding length: {len(sent_embs[0][1])}')

    # If you have to embed a large number of sentences
    # the embedding process can be embedded in batch:
    # sentences = ['This framework generates embeddings for each input sentence',
    # 'Sentences are passed as a list of string.', 
    # 'The quick brown fox jumps over the lazy dog.']
    # sentence_embeddings = model.encode(sentences)
    
    