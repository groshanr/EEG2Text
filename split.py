import pickle as pkl
import numpy as np

f = open('C:/Users/wadem/OneDrive - UCB-O365/EEG2Text/EEG2Text/Clone/x_train.pkl', 'rb')
x = pkl.load(f)
f.close()

f = open('C:/Users/wadem/OneDrive - UCB-O365/EEG2Text/EEG2Text/Clone/y_train.pkl', 'rb')
y = pkl.load(f)
f.close()

