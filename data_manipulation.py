import numpy as np

def shuffle_them(X,Y):
    tmp=np.concatenate((X,Y), axis =1)
    np.random.shuffle(tmp)
    [m, n] = np.shape(X)
    X=tmp[..., 0:n]
    Y=tmp[..., n]
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

def convert_cases_to_probability(X, Y):
    [m, n] = np.shape(X);
    [X_unique, indices] = np.unique(X, axis=0, return_inverse=True) 
    [m_unique, n] = np.shape(X_unique)
    print("# "+str(m_unique/float(m)*100)+" percent of the data is unique")
    Y_unique = np.zeros(np.shape(Y))
    for i in range(0, m_unique):
        repeatition = 0
        for j in range(0, m):
            if (i==indices[j]):
                repeatition += 1.0
                Y_unique[i] += float(Y[j])
        Y_unique[i] /= repeatition
    return X_unique, Y_unique
