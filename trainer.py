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

#from sklearn.preprocessing import StandardScaler
## read data

n_features = 3
[X, Y] = read_all_data("X_all.dat", "Y_all.dat")

X = X[:, 0:n_features]

print("initial mean: "+str(np.mean(X, axis = 0)))
print("initial std: "+str(np.std(X, axis = 0)))
X = mean_normalize(X)
print("mean after normalization: "+str(np.mean(X, axis = 0)))
print("std after normalization: "+str(np.std(X, axis = 0)))

[X, Y] = convert_cases_to_probability(X, Y)
[X, Y] = shuffle_them(X, Y)

[X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)

print(Y[1:10])
print(X[1:10])
model = Sequential()
model.add(Dense(15, input_dim=5, init= 'uniform' , activation= 'relu' ))
model.add(Dense(10, init= 'uniform' , activation= 'relu' ))
model.add(Dense(10, init= 'uniform' , activation= 'relu' ))
model.add(Dense(5, init= 'uniform' , activation= 'relu' ))
model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
model.fit(X, Y, nb_epoch=20, batch_size=100, validation_split=0.25)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicted = model.predict(X_eval, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
print((predicted[1:10]>0.5).astype(int))
print(Y_eval[1:10])
