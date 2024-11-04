import tensorflow as tf
import keras
import pandas as pd
import pickle as pkl
import gensim.downloader
import mne

#Load training data
f= open('mini_x_train.pkl', 'wb') 
train_x = pkl.load(f) 
f.close()

f= open('mini_y_train.pkl', 'wb') 
train_y = pkl.load(f) 
f.close()

#Load Keras Model
semantic_encoder = keras.saving.load_model("SemanticModel.keras")

#Compile
semantic_encoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CosineSimilarity(),
    metrics=[keras.metrics.MeanSquaredError(), keras.metrics.CosineSimilarity()]
)

#Train
semantic_encoder.fit(
    x = train_x,
    y = train_y,
    batch_size = 32,
    epochs = 10,
    validation_split = 0.2,
    verbose = 2
)

semantic_encoder.save('SemanticModule.keras')