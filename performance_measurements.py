import numpy as np
def performance(prediction, Y):
    
    all_one = np.ones(np.shape(Y))
    all_zero = np.zeros(np.shape(Y))
    print("# accuracy :"+str(1-np.sqrt(np.mean((prediction - Y)**2))))
    TP = float(sum(prediction*Y))
    Precision = TP/sum(prediction)
    Recall = TP/sum(Y)
    F1 = 2*(Recall * Precision) / (Recall + Precision)
    print("# if the alogrithm only would predict 0, accuracy :"+str(1-np.sqrt(np.mean((all_zero - Y)**2))))
    print("# precision :"+str(Precision))
    print("# recall :"+str(Recall))
    print("# F1 :"+str(F1))


def performance_multivalued(prediction, Y):
    correctly_predicted = (prediction==Y).astype(int)
    nr_correctly_predicted = float(sum(correctly_predicted))
    print("# accuracy :"+str(nr_correctly_predicted / np.size(Y)))

