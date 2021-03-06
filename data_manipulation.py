import numpy as np

def shuffle_them(X,Y):
    tmp=np.concatenate((X,Y), axis =1)
    np.random.shuffle(tmp)
    [m, n] = np.shape(X)
    X=tmp[:, 0:n]
    Y=tmp[:, n]
    #m = (len(Y));  # number of total samples
    Y = Y.reshape(m,1)
    return X,Y

def return_a_batch(batchid,batchsize,Xall,Yall):
    
    if ((batchid+1)*batchsize > np.size(Yall)): #Assuming Yall is a one dimensional array
        batchid=0
        print("sth went wrong in return a batch")
	    
    xbatch=Xall[batchid*batchsize:(batchid+1)*batchsize]
    ybatch=Yall[batchid*batchsize:(batchid+1)*batchsize]

    return xbatch, ybatch

def read_all_data(inputfile, outputfile):
    X = np.loadtxt(fname=inputfile).astype(np.float32)
    Y = np.loadtxt(fname=outputfile).astype(np.float32)
    m = (len(Y));  # number of total samples
    Y = Y.reshape(m,1)
    print("shape of the features: " + str(np.shape(X)))
    print("shape of the y: " + str(np.shape(Y)))
    return X, Y

def slice_the_data(X, Y):
    [m, n] = np.shape(X);
    mtrain = int(0.8*m)
    X_train = X[0:mtrain, :]
    X_eval = X[mtrain:m, :]
    Y_train = Y[0:mtrain, :]
    Y_eval = Y[mtrain:m, :]
    return X_train, Y_train, X_eval, Y_eval

def mean_normalize(X):
    result=X-np.mean(X, axis=0)
    std = np.std(result, axis=0)
    stdp = np.transpose(std[:,np.newaxis])
    return result/stdp

def max_min_normalize(X):
    result = X - np.min(X, axis=0)
    std = np.max(result, axis=0) - np.min(result, axis=0)
    stdp = np.transpose(std[:,np.newaxis])
    return result/stdp

def convert_cases_to_probability(X, Y):
    [m, n] = np.shape(X);
    [X_unique, indices] = np.unique(X, axis=0, return_inverse=True) 
    [m_unique, n] = np.shape(X_unique)
    print("# "+str(m_unique/float(m)*100)+" percent of the data is unique")
    Y_unique = np.zeros(shape = (m_unique, 1))
    for i in range(0, m_unique):
        mask = indices==i
        repeatition = np.sum(mask)
        Y_sel = Y[mask]
        Y_unique[i] = np.sum(Y_sel)/repeatition
    return X_unique, Y_unique

def write_all_data(X, filename):
    np.savetxt(filename, X)
    print("# the data is written in "+filename)

def descritize(X, col, binwidth):
    X[:, col] = np.floor(X[:, col]/binwidth)

# This function is basically doing the same job as the above function with a difference that if an element of the X array has a value equal to the maximum of the array, then an epsilon = 10^-5 is substracted from it.
def descritize_with_max(X, col, binwidth):
    epsilon = 10**-5
    maxval = np.max(X[:, col])
    X[:, col] = X[:, col] - epsilon * (X[:, col] == maxval)
    X[:, col] = np.floor(X[:, col]/binwidth) 
