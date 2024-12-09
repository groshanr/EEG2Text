
import os
import mne
import numpy as np
import pandas as pd
import pickle as pkl
import gensim.downloader
import random

folder = '/scratch/alpine/mawa5935/EEG2Text/Data/Session0/'

empty = np.zeros(276)

x = []
y = []
OOV = []
vocabulary = {}
model = gensim.downloader.load('conceptnet-numberbatch-17-06-300')

f = open('POS_map.pkl', 'rb')
POS_map = pkl.load(f)
f.close()

for f in os.listdir(folder):
    filename = folder+f
    epochs = mne.read_epochs(filename)
    epochs_df = epochs.to_data_frame()
    metadata = epochs.metadata
    metadata['pos'] = metadata['pos'].astype('category')
    metadata['pos'] = metadata['pos'].map(POS_map)

    for s in np.unique(metadata['sent_ident']):
        sentence_meta = metadata[metadata['sent_ident']==s]
        sentence_eeg = epochs_df[epochs_df['epoch'].isin(sentence_meta.index)]
        sentence_x = np.zeros((32,9,11,276))
        sentence_y = np.zeros((32, 304))
        length = len(np.unique(sentence_eeg['epoch']))

        if length<5 or length>29:
            #skip sentences that are too long or too short
            continue
        
        i = 0

        for w in np.unique(sentence_eeg['epoch']):

            meta = sentence_meta.loc[w]

            #If word is in vocabulary, add the embedding and the relevant features
            word = '/c/en/'+ meta['word']

            vector = np.zeros((304))

            if model.has_index_for(word):
                vector[0:300] = model.get_vector(word, norm=True)
                vector[301] = meta['pos']
                vector[302] = meta['len']
                vector[303] = meta['freq']
            elif any(char.isdigit() for char in word):
                #most of the out of vocabulary words are numbers, so since we care about semantics, we can just make the correct be the word "number"
                vector[0:300] = model.get_vector('/c/en/number', norm=True)
                vector[301] = meta['pos']+1
                vector[302] = meta['len']
                vector[303] = meta['freq']
            else:
                #other OOV words will result in the sentence being omitted
                OOV.append(word)
                break

            if word not in vocabulary.keys():
                vocabulary[word] = vector[0:300]

            sentence_y[i] = vector
            
            e = sentence_eeg[sentence_eeg['epoch']==w]

            epoch_reshape = np.array([[empty,   empty,   empty,      empty,      e['Fp1'],  empty,   e['Fp2'],  empty,   empty,      empty,     empty],
                                    [empty,   empty,   empty,      e['AF7'],   e['AF3'],  e['AFz'],e['AF4'],  e['AF8'],empty,      empty,     empty],
                                    [empty,   e['F7'], e['F5'],    e['F3'],    e['F1'],   e['Fz'], e['F2'],   e['F4'], e['F6'],    e['F8'],   empty],
                                    [e['FT9'],e['FT7'],e['FC5'],   e['FC3'],   e['FC1'],  e['FCz'],e['FC2'],  e['FC4'],e['FC6'],   e['FT8'],  e['FT10']],
                                    [empty,   e['T7'], e['C5'],    e['C3'],    e['C1'],   e['Cz'], e['C2'],   e['C4'], e['C6'],    e['T8'],   empty],
                                    [e['TP9'],e['TP7'],e['CP5'],   e['CP3'],   e['CP1'],  e['CPz'],e['CP2'],  e['CP4'],e['CP6'],   e['TP8'],  e['TP10']],
                                    [empty,   e['P7'], e['P5'],    e['P3'],    e['P1'],   e['Pz'], e['P2'],   e['P4'], e['P6'],    e['P8'],   empty],
                                    [empty,   empty,   empty,      e['PO7'],   e['PO3'],  e['POz'],e['PO4'],  e['PO8'],empty,      empty,     empty],
                                    [empty,   empty,   empty,      empty,      e['O1'],   e['Oz'], e['O2'],   empty,   empty,      empty,     empty]])

            sentence_x[i] = epoch_reshape

            i = i + 1


        x.append(sentence_x)
        y.append(sentence_y)



f= open('x.pkl', 'wb') 
pkl.dump(x, f)
f.close()

f= open('y.pkl', 'wb') 
pkl.dump(y, f)
f.close()

n = len(x)
idx = list(range(n))
random.shuffle(idx)

train_idx = idx[0:int(np.round(n*0.9))]
test_idx = idx[int(np.round(n*0.9)):n]

x_train = []
x_test = []

y_train = []
y_test = []

for i in range(n):
    if i in train_idx:
        x_train.append(x[i])
        y_train.append(y[i])
    elif i in test_idx:
        x_test.append(x[i])
        y_train.append(y[i]) 
    else:
        print('error at ', i)
        print(x[i].shape)


f= open('x_train.pkl', 'wb') 
pkl.dump(x_train, f)
f.close()

f= open('y_train.pkl', 'wb') 
pkl.dump(y_train, f)
f.close()

f= open('x_test.pkl', 'wb') 
pkl.dump(x_test, f)
f.close()

f= open('y_test.pkl', 'wb') 
pkl.dump(y_test, f)
f.close()
