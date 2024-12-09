import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import pickle as pkl
import gensim.downloader
import mne

#Load training data
f= open('train_x.pkl', 'rb') 
train_x = pkl.load(f) 
f.close()

f= open('train_y.pkl', 'rb') 
train_y = pkl.load(f) 
f.close()


CNN_input = keras.Input(shape=(32, 9, 11, 276, 1), batch_size=32, name = 'SentenceInput')

#Conv Block 1
x = layers.TimeDistributed(layers.BatchNormalization(name = 'BatchNorm1'))(CNN_input)
x = layers.TimeDistributed(layers.Conv3D(16, kernel_size=(5,5, 1), strides=1, padding = 'same', activation='relu', name = 'Conv1'))(x)

#Find time attention for each channel
time_dense1 = layers.TimeDistributed(layers.Dense(32, activation = 'relu', name = 'TimeDense1_1'))
time_dense2 = layers.TimeDistributed(layers.Dense(16, activation = 'relu', name = 'TimeDense1_2'))

fx_max = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (9, 11, 1), name = 'TimeMaxPool1'))(x)
fx_max = time_dense1(fx_max)
fx_max = time_dense2(fx_max)

fx_avg = layers.TimeDistributed(layers.AveragePooling3D(pool_size= (9, 11, 1), name = 'TimeAvgPool1'))(x)
fx_avg = time_dense1(fx_avg)
fx_avg = time_dense2(fx_avg)

time_attention = layers.Add(name = 'TimeAdd1')([fx_avg, fx_max])
time_attention = layers.Activation('sigmoid', name = 'TimeSigmoid1')(time_attention)

fx = layers.Multiply(name = 'TimeAttnCreate1')([time_attention, x])

#Find space attention for each channel
fx_max_space = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (1,1,276), name = 'SpaceMaxPool1'))(fx)
fx_avg_space = layers.TimeDistributed(layers.AveragePooling3D(pool_size = (1,1,276), name = 'SpaceAvgPool1'))(fx)
space_attention = layers.Concatenate(axis = 3, name = 'SpaceConcat1')([fx_avg_space, fx_max_space])
space_attention = layers.TimeDistributed(layers.Conv3D(filters = 1, kernel_size = (2,2,1), strides = (1,2,1), padding= 'same', activation='sigmoid', name = 'SpaceConvAttn1'))(space_attention)

fx = layers.Multiply(name = 'SpaceAttnCreate1')([space_attention, fx])

#Apply Time and Space attention
block1out = layers.Add(name = 'AttnApply1')([fx, x])

#Conv Block 2

x = layers.TimeDistributed(layers.BatchNormalization(name = 'BatchNorm2'))(block1out)
x = layers.TimeDistributed(layers.Conv3D(32, kernel_size=(5,5, 1), strides=1, padding = 'same', activation='relu', name = 'Conv2'))(x)

#Find time attention for each channel
time_dense3 = layers.TimeDistributed(layers.Dense(64, activation = 'relu', name = 'TimeDense2_1'))
time_dense4 = layers.TimeDistributed(layers.Dense(32, activation = 'relu', name = 'TimeDense2_2'))

fx_max = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (9, 11, 1), name = 'TimeMaxPool2'))(x)
fx_max = time_dense3(fx_max)
fx_max = time_dense4(fx_max)

fx_avg = layers.TimeDistributed(layers.AveragePooling3D(pool_size= (9, 11, 1), name = 'TimeAvgPool2'))(x)
fx_avg = time_dense3(fx_avg)
fx_avg = time_dense4(fx_avg)

time_attention = layers.Add(name = 'TimeAdd2')([fx_avg, fx_max])
time_attention = layers.Activation('sigmoid', name = 'TimeSigmoid2')(time_attention)

fx = layers.Multiply(name = 'TimeAttnCreate2')([time_attention, x])

#Find space attention for each channel
fx_max_space = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (1,1,276), name = 'SpaceMaxPool2'))(fx)
fx_avg_space = layers.TimeDistributed(layers.AveragePooling3D(pool_size = (1,1,276), name = 'SpaceAvgPool2'))(fx)
space_attention = layers.Concatenate(axis = 3, name = 'SpaceConcat2')([fx_avg_space, fx_max_space])
space_attention = layers.TimeDistributed(layers.Conv3D(filters = 1, kernel_size = (2,2,2), strides = (1,2,1), padding= 'same', activation='sigmoid', name = 'SpaceConvAttn2'))(space_attention)

fx = layers.Multiply(name = 'SpaceAttnCreate2')([space_attention, fx])
block2out = layers.Add(name = 'TimeSpaceAttnApply2')([fx, x])

#Conv Block 3
x = layers.TimeDistributed(layers.BatchNormalization(name = 'BatchNorm3'))(block2out)
x = layers.TimeDistributed(layers.Conv3D(64, kernel_size=(5,5, 1), strides=1, padding = 'same', activation='relu', name = 'Conv3'))(x)

time_dense5 = layers.TimeDistributed(layers.Dense(32, activation = 'relu', name = 'TimeDense3_1'))
time_dense6 = layers.TimeDistributed(layers.Dense(64, activation = 'relu', name = 'TimeDense3_2'))

fx_max = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (9, 11, 1), name = 'TimeMaxPool3'))(x)
fx_max = time_dense5(fx_max)
fx_max = time_dense6(fx_max)

fx_avg = layers.TimeDistributed(layers.AveragePooling3D(pool_size= (9, 11, 1), name = 'TimeAvgPool3'))(x)
fx_avg = time_dense5(fx_avg)
fx_avg = time_dense6(fx_avg)

time_attention = layers.Add(name = 'TimeAdd3')([fx_avg, fx_max])
time_attention = layers.Activation('sigmoid', name = 'TimeSigmoid3')(time_attention)

fx = layers.Multiply(name = 'TimeAttnCreate3')([time_attention, x])

#Find space attention for each channel
fx_max_space = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (1,1,276), name = 'SpaceMaxPool3'))(fx)
fx_avg_space = layers.TimeDistributed(layers.AveragePooling3D(pool_size = (1,1,276), name = 'SapceAvgPool3'))(fx)
space_attention = layers.Concatenate(axis = 3, name = 'SpaceConcat3')([fx_avg_space, fx_max_space])
space_attention = layers.TimeDistributed(layers.Conv3D(filters = 1, kernel_size = (2,2,2), strides = (1,2,1), padding= 'same', activation='sigmoid', name = 'SpaceConv3'))(space_attention)

fx = layers.Multiply(name = 'SpaceAttnCreate3')([space_attention, fx])
block3out = layers.Add(name = 'TimeSpaceAttnApply3')([fx, x])

#Conv Block 4

x = layers.TimeDistributed(layers.BatchNormalization(name = 'BatchNorm4'))(block3out)
x = layers.TimeDistributed(layers.Conv3D(32, kernel_size=(5,5, 1), strides=1, padding = 'same', activation='relu', name = 'Conv4'))(x)

time_dense7 = layers.TimeDistributed(layers.Dense(64, activation = 'relu', name = 'TimeDense4_1'))
time_dense8 = layers.TimeDistributed(layers.Dense(32, activation = 'relu', name = 'TimeDense4_2'))

fx_max = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (9, 11, 1), name='TimeMaxPool4'))(x)
fx_max = time_dense7(fx_max)
fx_max = time_dense8(fx_max)

fx_avg = layers.TimeDistributed(layers.AveragePooling3D(pool_size= (9, 11, 1), name = 'TimeAvgPool4'))(x)
fx_avg = time_dense7(fx_avg)
fx_avg = time_dense8(fx_avg)

time_attention = layers.Add(name = 'TimeAdd4')([fx_avg, fx_max])
time_attention = layers.Activation('sigmoid', name = 'TimeSigmoid4')(time_attention)

fx = layers.Multiply(name = 'TimeAttnCreate4')([time_attention, x])

#Find space attention for each channel
fx_max_space = layers.TimeDistributed(layers.MaxPooling3D(pool_size = (1,1,276), name = 'SpaceMaxPool4'))(fx)
fx_avg_space = layers.TimeDistributed(layers.AveragePooling3D(pool_size = (1,1,276), name = 'SpaceAvgPool4'))(fx)
space_attention = layers.Concatenate(axis = 3, name = 'SpaceConcat4')([fx_avg_space, fx_max_space])
space_attention = layers.TimeDistributed(layers.Conv3D(filters = 1, kernel_size = (2,2,2), strides = (1,2,1), padding= 'same', activation='sigmoid', name = 'SpaceConv4'))(space_attention)

fx = layers.Multiply(name = 'SpaceAttnCreate4')([space_attention, fx])
block4out = layers.Add(name = 'TimeSpaceAttnApply4')([fx, x])

# Final word vector maker
x = layers.TimeDistributed(layers.MaxPool3D(pool_size = (3,3,25), name = 'FinalMaxPool'))(block4out)
x = layers.TimeDistributed(layers.Flatten(name = 'Flatten'))(x)
x = layers.TimeDistributed(layers.Dense(64, activation='relu', name = 'FinalDense1'))(x)
x = layers.TimeDistributed(layers.Dense(32, activation='relu', name = 'FinalDense2'))(x)
x = layers.TimeDistributed(layers.Dense(64, activation='relu', name = 'FinalDense3'))(x)
x = layers.TimeDistributed(layers.Dense(16, name = 'PosOutput', activation='softmax'))(x)
CNN_output = layers.TimeDistributed(layers.Normalization(axis=-1, name = 'Pos Normalization'))(x)

pos_model = keras.Model(CNN_input, CNN_output)

#Compile
pos_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy,
    metrics=[keras.metrics.MeanSquaredError(), keras.metrics.CosineSimilarity()]
)

#Train
pos_model.fit(
    x = train_x,
    y = train_y,
    batch_size = 32,
    epochs = 25,
    validation_split = 0.2,
    verbose = 2
)

semantic_encoder.save('SemanticModule.keras')