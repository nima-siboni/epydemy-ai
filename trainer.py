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
from performance_measurements import performance
from performance_measurements import performance_multivalued
from keras.utils import plot_model
from data_manipulation import descritize_with_max
from data_manipulation import descritize
## read data
[X, Y] = read_all_data("X_all_processed_3_features.dat", "Y_all_processed_3_features.dat")
m = 10000
X = X[0:m, :]
Y = Y[0:m, :]

[m, n_features] = np.shape(X)
[X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)

model = Sequential()
model.add(Dense(10, input_dim=n_features, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(5, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
#model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=50, batch_size=100)
scores = model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicted = model.predict(X_eval, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)


print("The predictions are coarse grained to high risk and low risk populations")
highrisk_criterion = 0.75
print("If the probability is larger than "+str(highrisk_criterion*100)+"% that individual is considered high risk")
coarse_prediction = (predicted > highrisk_criterion).astype(int)
coarse_Y = (Y_eval > highrisk_criterion).astype(int)
performance(coarse_prediction, coarse_Y)

#binwidth = 1.0/4.0
#descritize_with_max(predicted, 0, binwidth)
#descritize_with_max(Y_eval, 0, binwidth)
#performance_multivalued(predicted, Y_eval)

print(predicted[0:10,:])
print(Y_eval[0:10,:])



# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
