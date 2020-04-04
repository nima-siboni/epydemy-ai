from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from data_manipulation import shuffle_them 
from data_manipulation import return_a_batch
from data_manipulation import read_all_data
from data_manipulation import slice_the_data
from data_manipulation import mean_normalize
from data_manipulation import convert_cases_to_probability


## read data
[X, Y] = read_all_data("X_all_processed.dat", "Y_all_processed.dat")

[X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)

print(Y[1:10])
print(X[1:10])
model = Sequential()
model.add(Dense(15, input_dim=n_features, init= 'uniform' , activation= 'relu' ))
model.add(Dense(10, init= 'uniform' , activation= 'relu' ))
model.add(Dense(5, init= 'uniform' , activation= 'relu' ))
model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
model.fit(X, Y, nb_epoch=20, batch_size=100, validation_split=0.25)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicted = model.predict(X_eval, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
print(predicted[1:10])
print(Y_eval[1:10])
