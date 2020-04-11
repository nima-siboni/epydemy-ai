from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
from data_manipulation import write_all_data


# Network Topology search parameters
StartLayerN = 2
EndLayerN = 12
LayerStep = 2

StartNeuronN = 4
EndNeuronN = 14
NeuronStep = 2

NRepeat = 5  # repeat each case this many times (to get the best score)

## read data
[X, Y] = read_all_data("X_all_processed_3_features.dat", "Y_all_processed_3_features.dat")
m = 5000
X = X[0:m, :]
Y = Y[0:m, :]

[m, n_features] = np.shape(X)
# [X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)
X_train = X
Y_train  = Y

sizeLayer = np.arange(StartLayerN, EndLayerN, LayerStep).size
sizeNeuron = np.arange(StartNeuronN, EndNeuronN, NeuronStep).size
score_table = np.zeros([sizeLayer,sizeNeuron ])
i_layer = -1

Best_All_Score = 1e20
Best_All_Score_layer = 0
Best_All_Score_neuron = 0

for n_layer in range(StartLayerN,EndLayerN,LayerStep):
    i_layer +=1
    i_neuron = -1
    for n_neuron in range(StartNeuronN,EndNeuronN,NeuronStep):
        i_neuron +=1
        best_scores = 1e20
        for n_repeat in range(1,NRepeat+1):
            model = None
            model = Sequential()
            model.add(Dense(n_neuron, input_dim=n_features, kernel_initializer= 'random_uniform' , activation= 'relu' ))
            for nn_layer in range(0,n_layer): 
                model.add(Dense(n_neuron, kernel_initializer= 'random_uniform' , activation= 'relu' ))
            model.add(Dense(1, kernel_initializer= 'random_uniform' , activation= 'sigmoid' ))
            model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=['accuracy'])
            model.fit(X_train, Y_train, epochs=50, batch_size=100, verbose=0)
            scores = model.evaluate(X_train, Y_train, verbose=0)
            print("n_layer: %i, n_neuron: %i, repeat: %i, score: %.5f" % (n_layer, n_neuron, n_repeat, scores[0]) )
            if (scores[0] < best_scores): best_scores = scores[0]
        print("Best Score: %.5f" % (best_scores))
        score_table[i_layer,i_neuron] = best_scores
        if (best_scores<Best_All_Score):
            Best_All_Score = best_scores
            Best_All_Score_layer = n_layer
            Best_All_Score_neuron = n_neuron
        

print("Best All Score: %.5f with %i layers %i neurons" % (Best_All_Score,Best_All_Score_layer,Best_All_Score_neuron ))

write_all_data(score_table,"Scoretable.txt")

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(StartLayerN, EndLayerN, LayerStep)
Y = np.arange(StartNeuronN, EndNeuronN, NeuronStep)
X, Y = np.meshgrid(X, Y)
Z = score_table
        
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
