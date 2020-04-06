import numpy as np
from data_manipulation import read_all_data
from data_manipulation import convert_cases_to_probability
from data_manipulation import write_all_data
from data_manipulation import mean_normalize
from data_manipulation import max_min_normalize
from data_manipulation import shuffle_them 
from data_manipulation import descritize
# Read all data
# (0) nr_people_in_home 	
# (1) nr_people_at_work 	
# (2) nr_socialplaces 	
# (3) total_number of people in the socialing places 	
# (4) maximum number of people in socialing places 
[X, Y] = read_all_data("X_all.dat", "Y_all.dat")

# Lets discritize the total number of people that one person meets in all the social places
# This data is col = 4, the binwidth is chosen from the observation of the histogram of this feature
# col = 3
# binwidth = 75
# descritize(X, col, binwidth)
# print(X[0:10,:])
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

write_all_data(X, "X_all_processed_3_features.dat")
write_all_data(Y, "Y_all_processed_3_features.dat")

