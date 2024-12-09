import tensorflow as tf
import keras
import pandas as pd
import pickle as pkl
import gensim.downloader
import mne
import numpy as np
import json

f = open('x_train.pkl', 'rb')
x_train = pkl.load(f)
f.close()

f = open('y_train.pkl', 'rb')
y_train = pkl.load(f)
f.close()

batch_size = 32

over = np.mod(len(y_train), batch_size)
y_train = y_train[0:int(len(y_train)-over)]
x_train = x_train[0:int(len(x_train)-over)]

y_train_new = np.zeros((len(y_train), 32))
i = 0
for y in y_train:
    j = 0
    for w in y:
        y_train_new[i][j] = w[301]
        j = j+1
    i = i+1

y_train = y_train_new

x_train = np.array(x_train)

pos_model = keras.models.load_model("PosModel.keras")


pos_model.fit(
    x = x_train,
    y =  y_train,
    batch_size = batch_size,
    epochs = 1,
    validation_split = 0.1,
    callbacks = keras.callbacks.BackupAndRestore(
    "temp_pos/", save_freq=5, delete_checkpoint=True, save_before_preemption=False
)
)

pos_model.save('PosModel.keras')

#Evaluate

f = open('x_test.pkl', 'rb')
x_test = pkl.load(f)
f.close()

f = open('y_test.pkl', 'rb')
y_test = pkl.load(f)
f.close()

y_test_new = np.zeros((len(y_test), 32))
i = 0
for y in y_test:
    j = 0
    for w in y:
        y_test_new[i][j] = w[301]
        j = j+1
    i = i+1

y_test = y_test_new

x_test = np.array(x_test)

results = pos_model.evaluate(
    x = x_test,
    y = y_test,
    return_dict = True
    )

# Save the dictionary to a file
with open('pos-4real_results.txt', 'w') as f:
    json.dump(results, f)

