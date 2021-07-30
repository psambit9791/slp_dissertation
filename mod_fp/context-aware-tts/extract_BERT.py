from tqdm import tqdm
import numpy
from numpy import asarray
from numpy import save
import csv
import re
import os
import argparse
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
tqdm.pandas()

############################################################################################################
# Based on tutorial from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#sentence-vectors
#
#
#
#
##########################################################################################################


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_args():
    parser = argparse.ArgumentParser(description='This script takes '
                                                 'the annotations from Suni et al 2020 '
                                                 'and converts them to fastspeech.')
    parser.add_argument('path_csv_texts', metavar='N', type=str,
                        help='path to csv')
    parser.add_argument('outpath_token_embeddings', metavar='N', type=str,
                        help='path for results')
    parser.add_argument('outpath_utterance_embeddings', metavar='N', type=str,
                        help='output numpy files')
    args = parser.parse_args()

    return args.path_csv_texts, args.outpath_token_embeddings, args.outpath_utterance_embeddings


def tokenise(text):
    '''This function tokenises on white spaces'''

   # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tok_text = tokenizer.tokenize(text)
    inputs = tokenizer(text)
    print(len(tok_text))
    return(tok_text)


def make_sentence_ids(tokenised_text):

    ids = [1] * len(tokenised_text)

    return ids


def convert_to_indices(text):

    indices = tokenizer.convert_tokens_to_ids(text)

    return indices


def add_bert_symbols(text):


    bert_text = "[CLS] " + text + " [SEP]"

    return bert_text


def convert_punctuation(text):
    '''This function converts pseudo punctuation back to standard for tagging'''

    corrected = re.sub(r" xxperiodxx", r".", text)
    corrected = re.sub(r" xxcommaxx", r",", corrected)
    corrected = re.sub(r" xxexclamationxx", r"!", corrected)
    corrected = re.sub(r" xxquestionxx", r"?", corrected)

    return corrected


def get_bert_embeddings(tokenised_text):

    tokens_indices = tokenizer.convert_tokens_to_ids(tokenised_text)
    segments_ids = ids = [1] * len(tokenised_text)
    token_tensor = torch.tensor([tokens_indices])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )
    model.eval()
    with torch.no_grad():
        outputs = model(token_tensor, segments_tensors)

    return outputs


def get_token_embedding(hidden_states):
    '''Combining every layer into 1 tensor followed by removing
    the batch layer since it is single sentence processing'''

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    return token_embeddings


def get_sentence_embedding(hidden):
    '''Average the second last hidden layer of each token
     and calculate average'''

    token_vecs = hidden[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.numpy()
    return sentence_embedding


def get_word_embeddings(token_embeddings):
    '''Most work seems to talk about using last four layers.
    although task dependent, the later layers contain more
    contextual information, so will go for this
    Further, summing seems to be preferred since smaller in dim'''

    token_vecs_sum = []

    for token in token_embeddings:
        sum_vec = torch.sum(token[-5:], dim=0)
        sum_vec = sum_vec.numpy()
        token_vecs_sum.append(sum_vec)
    #print(type(sum_vec))
    #token_vecs_array = numpy.array(token_vecs_sum)
    token_vecs_sum = numpy.stack(token_vecs_sum, axis=0 )
    print(token_vecs_sum.shape)

    return(token_vecs_sum)


def get_hidden_states(model_output):

    hidden = model_output[2]

    return hidden


def make_tokenise_files(data_path, outpath_token_embed, outpath_sentence_embed):
    '''This function loads a csv of form File | Text and performs tagging'''

    print("Reading in csv of files and texts\n")
    df = pd.read_csv(data_path, sep='|', quoting=csv.QUOTE_NONE, error_bad_lines=False, header=None)
    #df.columns = ['Filename', 'Text']
    df.columns = ['Filename', 'text1' ,'text2', 'Text']

    print("Converting Punctuation words back to symbols...\n")
    df['Text_normalised'] = df['Text'].progress_apply(convert_punctuation)
    print("Adding BERT specific tokens\n")
    df['Text_bert_format'] = df['Text_normalised'].progress_apply(add_bert_symbols)
    print("Tokenising to BERT format\n")
    df['Tokenisation'] = df['Text_bert_format'].progress_apply(tokenise)
    print("Converting tokens to indices\n")
    df['Token_indices'] = df['Tokenisation'].progress_apply(convert_to_indices)
    print("Creating single sentence IDs\n")
    df['Sentence_IDs'] = df['Tokenisation'].progress_apply(make_sentence_ids)
    #df['Token_indices_tensors'] = df['Token_indices'].progress_apply(torch.tensor)
    #df['Sentence_IDs_tensors'] = df['Sentence_IDs'].progress_apply(torch.tensor)
    #df['Bert_embeddings'] = df.progress_apply(lambda x: get_bert_embeddings(x['Token_indices_tensors'], x['Sentence_IDs_tensors']), axis=1)
    print("Getting BERT evaluation of utterance...")
    df['Bert_output'] = df['Tokenisation'].progress_apply(get_bert_embeddings)
    print("Extracting hidden states...\n")
    df['Hidden_states'] = df['Bert_output'].progress_apply(get_hidden_states)
    print("Converting to single tensor...\n")
    df['Token_embeddings'] = df['Hidden_states'].progress_apply(get_token_embedding)
    print("Creating Token level embeddings...\n")
    df['Token_arrays'] = df['Token_embeddings'].progress_apply(get_word_embeddings)
    print("Creating sentence level embeddings..")
    df['Sentence_embeddings'] = df['Hidden_states'].progress_apply(get_sentence_embedding)
    print(df)
    # write token level nump arrays to folder
    print("Saving token embeddings to " + outpath_token_embed)
    token_embed = pd.Series(df.Token_arrays.values,index=df.Filename).to_dict()
    for key, value in tqdm(token_embed.items()):
        save(outpath_token_embed + key  + '.npy', value)

    # write token level nump arrays to folder
    print("Saving token embeddings to " + outpath_sentence_embed)
    sentence_embed = pd.Series(df.Sentence_embeddings.values,index=df.Filename).to_dict()
    for key, value in tqdm(sentence_embed.items()):
        save(outpath_sentence_embed + key  + '.npy', value)


def make_word_arrays(data_path, outpath_token_embed):

    print("Reading in csv of files and texts\n")
    df = pd.read_csv(data_path, sep='|', quoting=csv.QUOTE_NONE, error_bad_lines=False, header=None)
    df.columns = ['Filename', 'text1' ,'text2', 'Text']
    #df.columns = ['Filename', 'Text']

    print("Converting Punctuation words back to symbols...\n")
    df['Text_normalised'] = df['Text'].progress_apply(convert_punctuation)
    print("Adding BERT specific tokens\n")
    df['Text_bert_format'] = df['Text_normalised'].progress_apply(add_bert_symbols)
    print("Tokenising to BERT format\n")
    df['Tokenisation'] = df['Text_bert_format'].progress_apply(tokenise)
    df['Token_arrays'] = df['Tokenisation'].progress_apply(make_numpy_arrays)
    make_numpy_files(df, outpath_token_embed)
    print(df)

def make_numpy_files(df, outpath_token_embed):
    '''Takes df of texts and returns individual .lab files'''
    x = pd.Series(df.Token_arrays.values,index=df.Filename).to_dict()

    for key, value in tqdm(x.items()):
        save(outpath_token_embed + key  + '_token_index.npy', value)

def make_numpy_arrays(tokens):
    '''Takes df of texts and returns individual .lab files'''
    token_arrays = numpy.asarray(tokens, dtype=str)
    print(token_arrays.shape)
    return token_arrays

if __name__ == '__main__':

    only_make_arrays = False
    data_path, outpath_token_embed, outpath_sentence_embed = load_args()
    if not only_make_arrays:
        make_tokenise_files(data_path, outpath_token_embed, outpath_sentence_embed)
        make_word_arrays(data_path, outpath_token_embed)
    if only_make_arrays:
        make_word_arrays(data_path, outpath_token_embed)
