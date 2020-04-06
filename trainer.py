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
from keras.utils import plot_model
from data_manipulation import descritize
## read data
[X, Y] = read_all_data("X_all_processed_3_features.dat", "Y_all_processed_3_features.dat")
#Y = (Y>0.5).astype(int)
n_features = 3
X=X[:, 0:n_features]
[X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)

model = Sequential()
model.add(Dense(10, input_dim=n_features, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(5, kernel_initializer= 'uniform' , activation= 'relu' ))
#model.add(Dense(3, init= 'uniform' , activation= 'relu' ))
model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
#model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=10, batch_size=100)
scores = model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicted = model.predict(X_eval, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

print("The predictions are coarse grained to high risk and low risk populations")
coarse_prediction = (predicted>0.5).astype(int)
coarse_Y = (Y_eval>0.5).astype(int)
performance(coarse_prediction, coarse_Y)


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
