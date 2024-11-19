#Creating the 3D dataset
import os
import mne
import numpy as np
import pandas as pd
import pickle as pkl
import gensim.downloader

folder = ''
empty = np.zeros(276)
sent_end = np.random.rand(9,11,276)
sent_empty = np.zeros((9,11,276))
pad = np.zeros(300)
special_characters = '!@#$%^&*()_+-=[],."\''

x = []
y = []
OOV = []
vocabulary = {}
model = gensim.downloader.load('conceptnet-numberbatch-17-06-300')

for f in os.listdir(folder):
    epochs = mne.read_epochs(f)
    epochs_df = epochs.to_data_frame()
    metadata = epochs.metadata

    for s in np.unique(metadata['sent_ident']):
        sentence_meta = metadata[metadata['sent_ident']==s]
        sentence_eeg = epochs_df[epochs_df['epoch'].isin(sentence_meta.index)]
        sentence_x = []
        sentence_y = []
        length = len(np.unique(sentence_eeg['epoch']))

        if length<5 or length>30:
            #skip sentences that are longer than 30 words or less than 5 words
            continue
        
        for w in np.unique(sentence_eeg['epoch']):
            e = sentence_eeg[sentence_eeg['epoch']==w]
            meta = sentence_meta.loc[w]
            epoch_reshape = np.array([[empty,   empty,   empty,      empty,      e['Fp1'],  empty,   e['Fp2'],  empty,   empty,      empty,     empty],
                                    [empty,   empty,   empty,      e['AF7'],   e['AF3'],  e['AFz'],e['AF4'],  e['AF8'],empty,      empty,     empty],
                                    [empty,   e['F7'], e['F5'],    e['F3'],    e['F1'],   e['Fz'], e['F2'],   e['F4'], e['F6'],    e['F8'],   empty],
                                    [e['FT9'],e['FT7'],e['FC5'],   e['FC3'],   e['FC1'],  e['FCz'],e['FC2'],  e['FC4'],e['FC6'],   e['FT8'],  e['FT10']],
                                    [empty,   e['T7'], e['C5'],    e['C3'],    e['C1'],   e['Cz'], e['C2'],   e['C4'], e['C6'],    e['T8'],   empty],
                                    [e['TP9'],e['TP7'],e['CP5'],   e['CP3'],   e['CP1'],  e['CPz'],e['CP2'],  e['CP4'],e['CP6'],   e['TP8'],  e['TP10']],
                                    [empty,   e['P7'], e['P5'],    e['P3'],    e['P1'],   e['Pz'], e['P2'],   e['P4'], e['P6'],    e['P8'],   empty],
                                    [empty,   empty,   empty,      e['PO7'],   e['PO3'],  e['POz'],e['PO4'],  e['PO8'],empty,      empty,     empty],
                                    [empty,   empty,   empty,      empty,      e['O1'],   e['Oz'], e['O2'],   empty,   empty,      empty,     empty]])
            sentence_x.append(epoch_reshape)
            word = '/c/en/'+ meta['word']
            if model.has_index_for(word):
                emb = model.get_vector(word, norm=True)
            elif any(char.isdigit() for char in word):
                #most of the out of vocabulary words are numbers, so since we care about semantics, we can just make the correct be the word "number"
                emb = model.get_vector('/c/en/number', norm=True)
            elif any(char in special_characters for char in word):
                #other ones are just punctuation, so we can just use the word "punctuation"
                emb = model.get_vector('/c/en/punctuation', norm=True)
            else:
                #other OOV words will have the correct embedding be zeros 
                OOV.append(word)
                emb = pad
            sentence_y.append(emb)
            if word not in vocabulary.keys():
                vocabulary[word] = emb
        sentence_x.append(sent_end)
        sentence_y.append(pad)
        for i in range(31-length):
            sentence_x.append(sent_empty)
            sentence_y.append(pad)
        x.append(sentence_x)
        y.append(sentence_y)

train_x = x[0:int(np.floor(len(x)*0.9)), :, :, :, :]
test_x = x[int(np.ceil(len(x)*0.9):len(x)), :, :, :, :]

train_y = y[0:int(np.floor(len(y)*0.9)), :, :]
test_y = y[int(np.ceil(len(y)*0.9)), :, :]

f= open('train_x.pkl', 'wb') 
train_x = pkl.dump(f)
f.close()

f= open('train_y.pkl', 'wb') 
train_y = pkl.dump(f)
f.close()

f= open('test_x.pkl', 'wb') 
test_x = pkl.dump(f)
f.close()

f= open('test_y.pkl', 'wb') 
test_y = pkl.dump(f)
f.close()

