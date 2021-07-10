### Luke's Embedding Demo

import csv
import numpy as np

import torch
from transformers import BertTokenizer, BertModel

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
    # pip install spacy-transformers
    # python -m spacy download en_trf_bertbaseuncased_lg

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

        # If you want to embed the sentences separately
        # comment out the previous line and uncomment
        # this bit:
        # doc = nlp(text)
        # for j, sent in enumerate(doc.sents):
        #     # print(j, sent)
        #     sent_embs.append((sent, model.encode(sent)))

    
    print(f'sent_embs: {sent_embs[0][0]} embedding length: {len(sent_embs[0][1])}')

    # sentences = ['This framework generates embeddings for each input sentence',
    # 'Sentences are passed as a list of string.', 
    # 'The quick brown fox jumps over the lazy dog.']
    
    # sentence_embeddings = model.encode(sentences)
    
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
    # # inp_list = tokenizer.tokenize("Hello, my dog is cute")
    # # print(f'\n\nINPUTS LIST: {inp_list}\n\n')

    # # print(f
    # # 'help(tokenizer) {help(tokenizer)}')

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(f'\n\nINPUTS IDS: {inputs["input_ids"]}\n\n')
    # print(f'\n\nINPUT WORDS: {tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())}\n\n')
    # outputs = model(**inputs)

    # print(f'\n\ndir(outputs): {dir(outputs)}\n\n')
    # # print(f'\n\noutputs: {outputs}\n\n')
    # # print(f'\n\nhelp(outputs): {help(outputs)}\n\n')

    # # BERT WORD EMBEDDINGS
    # last_hidden_states = outputs.last_hidden_state
    # print(f'Last hidden states shape: {last_hidden_states.shape}')
    # print(f'Last hidden states: {last_hidden_states}')
    # hello_tensor = last_hidden_states[0][1]
    
    # hidden_states = outputs.hidden_states
    # print(f'\nHidden states: {len(hidden_states)} {hidden_states[0].shape}')
    
    # pooler_output = outputs.pooler_output
    # print(f'\npooler_output: {pooler_output.shape}')

    # print(f'\n\nmodel: {model}\n\n')
    
    # # BERT WORD EMBEDDINGS
    # inputs = tokenizer("Cute my dog Hello is", return_tensors="pt")
    # outputs = model(**inputs)
    # print(f'\n\nINPUTS IDS: {inputs["input_ids"]}\n\n')
    # print(f'\n\nINPUT WORDS: {tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())}\n\n')
    # last_hidden_states = outputs.last_hidden_state
    # print(f'Last hidden states shape: {last_hidden_states.shape}')
    # print(f'Last hidden states: {last_hidden_states}')
    # hello_tensor_2 = last_hidden_states[0][4]

    # if torch.equal(hello_tensor, hello_tensor_2):
    #     print(f'Both hello tensors are equal: {hello_tensor[:3]} and {hello_tensor_2[:3]}')
    # else:
    #     print(f'Both hello tensors are different: {hello_tensor[:3]} and {hello_tensor_2[:3]}')

    # convert_ids_to_tokens()
    # encode() Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))

    ## spaCy code
    ## simpler, but has versioning issues
    # nlp = spacy.load("en_trf_bertbaseuncased_lg")

    # apple1 = nlp("Apple shares rose on the news.")
    # apple2 = nlp("Apple sold fewer iPhones this quarter.")
    # apple3 = nlp("Apple pie is delicious.")

    # # sentence similarity
    # print(apple1.similarity(apple2)) #0.69861203
    # print(apple1.similarity(apple3)) #0.5404963

    # # sentence embeddings
    # print(apple1.vector)  # or apple1.tensor.sum(axis=0)

    # print(f'\n\ndir(apple1): {dir(apple1)}')


    