from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_manipulation import shuffle_them 
from data_manipulation import return_a_batch
from data_manipulation import read_all_data
from data_manipulation import slice_the_data

## read data

[X, Y] = read_all_data("X_all.dat", "Y_all.dat")

[X, Y] = shuffle_them(X, Y)

[X_train, Y_train, X_eval, Y_eval] = slice_the_data(X, Y)


# Parameters
learning_rate_Adam = 0.01
num_steps = 50000
batchsize = 100
display_step = 100
shuffling_time=10000
# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
n_hidden_3 = 10 # 2nd layer number of neurons
n_hidden_4 = 10 # 2nd layer number of neurons
#n_hidden_5 = 10 # 2nd layer number of neurons

num_input = 5 # NIn
num_output = 1 # NOut


# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# # Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
#    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
   'out': tf.Variable(tf.random_normal([n_hidden_4, num_output]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
#    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
   'out': tf.Variable(tf.random_normal([num_output]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer 
   layer_1 = tf.math.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
   layer_2 = tf.math.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
   layer_3 = tf.math.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
   layer_4 = tf.math.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
#   layer_5 = tf.math.sigmoid(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
   out_layer=tf.math.sigmoid(tf.add(tf.matmul(layer_4, weights['out']), biases['out']))
   return out_layer

# Construct model
logits = neural_net(X)
prediction = logits
# Define loss and optimizer
#loss_op = tf.nn.l2_loss( tf.subtract(logits,Y))
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_Adam, beta2=0.999, beta1=0.9, epsilon=1e-4).minimize(loss_op)
#train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_GD).minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
batchid=0

with tf.Session() as sess:
    # Run the initializer
   sess.run(init)
   for step in range(1, num_steps):
      
           # Shuffle the whole data
      if (step%shuffling_time == 0):
         [X_train,Y_train] = shuffle_them(X_train, Y_train)
	# Create the batch for this step
      [Xbatch, Ybatch] = return_a_batch(batchid, batchsize, X_train, Y_train)
      batchid = batchid + 1
      if (batchid + num_input > np.size(Y_train) / batchsize):
         batchid = 0
      # Run for this batch
      sess.run(train_op, feed_dict = {X: Xbatch, Y: Ybatch})
      if step % display_step == 0 or step == 1:
      # Calculate batch loss and accuracy
         loss = sess.run(loss_op, feed_dict = {X: Xbatch, Y: Ybatch})
         print("Step " + str(step) + ", Minibatch Loss= " +  str(loss))
         
   print("Optimization Finished!")
    


   prediction = sess.run(neural_net(X_eval))
   accuracy = sess.run(loss_op, feed_dict = {X: X_eval, Y: Y_eval })
   print(prediction)
   print(Y_eval)
   [m_eval, n_eval] = np.shape(Y_eval);
   print(" Evaluation starts with "+str(m_eval)+" samples.")
   print(" The obtained accuracy is " + "{:.4e}".format(accuracy))
