import numpy as np
from data_manipulation import read_all_data
from data_manipulation import convert_cases_to_probability
from data_manipulation import write_all_data
from data_manipulation import mean_normalize
from data_manipulation import max_min_normalize
from data_manipulation import shuffle_them 

# Read all data
[X, Y] = read_all_data("X_all.dat", "Y_all.dat")

# Only the first n_features are used.
n_features = 3
X = X[:, 0:n_features]


print("initial mean: "+str(np.mean(X, axis = 0)))
print("initial std: "+str(np.std(X, axis = 0)))
X = max_min_normalize(X)
print("mean after normalization: "+str(np.mean(X, axis = 0)))
print("std after normalization: "+str(np.std(X, axis = 0)))

[X, Y] = convert_cases_to_probability(X, Y)

[X, Y] = shuffle_them(X, Y)

write_all_data(X, "X_all_processed.dat")
write_all_data(Y, "Y_all_processed.dat")

