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

# y_train = y_train[0:64]
# x_train = x_train[0:64]

y_train_new = np.zeros((len(y_train), 32))
i = 0
for y in y_train:
    j = 0
    for w in y:
        y_train_new[i][j] = w[303]
        j = j+1
    i = i+1

y_train = y_train_new

x_train = np.array(x_train)

freq_model = keras.models.load_model("FrequencyModel.keras")


freq_model.fit(
    x = x_train,
    y =  y_train,
    batch_size = batch_size,
    epochs = 5,
    validation_split = 0.1,
    callbacks = keras.callbacks.BackupAndRestore(
    "temp_freq/", save_freq=5, delete_checkpoint=True, save_before_preemption=False)
)

freq_model.save('FrequencyModel.keras')

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
        y_test_new[i][j] = w[303]
        j = j+1
    i = i+1

y_test = y_test_new

x_test = np.array(x_test)

# x_test = x_test[0:32]
# y_test = y_test[0:32]

results = freq_model.evaluate(
    x = x_test,
    y = y_test,
    return_dict = True
    )

# Save the dictionary to a file
with open('frequency_results.txt', 'w') as f:
    json.dump(results, f)

