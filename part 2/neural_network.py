import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=False):

    #IMPLEMENT HERE
    losses = []
    for i in range(epoch):
        #print("Epoch %d" % i)
        secondrange = len(x_train)/200
        losstotal = 0

        if shuffle :
            state = np.random.get_state()
            np.random.shuffle(x_train) 
            np.random.set_state(state) #this should shuffle them in parallel
            np.random.shuffle(y_train)

        for j in range(secondrange):
            X = x_train[j*200:(j+1)*200]
            y = y_train[j*200:(j+1)*200]
            losstemp, w1, w2, w3, w4, b1, b2, b3, b4 = four_nn(X, w1, w2, w3, w4, b1, b2, b3, b4, y, test=False)
            losstotal += losstemp
        #print(losstemp)
        losses.append(losstotal)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    classification = np.zeros(len(x_test))
    #print(len(x_test))
    
    classification = four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, True)
    avg_class_rate = (classification == y_test).mean()
    y_unique, y_counts = np.unique(y_test, return_counts=True)
    for i in range(len(x_test)) :
        if classification[i] == y_test[i] :
            class_rate_per_class[classification[i]] += 1
    for i in range(num_classes):
        class_rate_per_class[i] /= y_counts[i]
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(X, w1, w2, w3, w4, b1, b2, b3, b4, y, test):
    
    Z1, acache1 = affine_forward(X, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, w2, b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, w3, b3)
    A3, rcache3 = relu_forward(Z3)
    F,  acache4 = affine_forward(A3, w4, b4)

    if test == True:
        classifications = []
        for i in range(len(F)) :
            classifications.append(np.argmax(F[i]))
        #print(len(classifications))
        return classifications

    loss, dF = cross_entropy(F, y)

    dA3, dw4, db4 = affine_backward(dF,  acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dw3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dw2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dw1, db1  = affine_backward(dZ1, acache1)
    w1 -= (0.1)*dw1
    w2 -= (0.1)*dw2
    w3 -= (0.1)*dw3
    w4 -= (0.1)*dw4
    b1 -= (0.1)*db1
    b2 -= (0.1)*db2
    b3 -= (0.1)*db3
    b4 -= (0.1)*db4

    return loss, w1, w2, w3, w4, b1, b2, b3, b4

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
	#Inputs: A (data with size n,d)
	#		 W (weights with size d, d')
	#		 b (bias with size d')
	#
	#Performs: Z_(i,j) = sum(k=0, d-1, A_(i,k)*W_(k,j) + b_(j))
	#
	#Outputs: Z (affine output with size n,d')
	#		  Cache (tuple of the original inputs)

	Z = None
	
	N = A.shape[0]
	D = np.prod(A.shape[1:])
	A2 = np.reshape(A, (N,D))
	Z = np.dot(A2, W) + b

	cache = (A, W, b)
	
	return Z, cache

def affine_backward(dZ, cache):
    """
    Inputs: dZ
            cache
    Outputs: dA
             dW
             dB
    """
  
    A, W, b = cache
    
    N = A.shape[0]
    D = np.prod(A.shape[1:])
    A2 = np.reshape(A, (N,D))

    dA2 = np.dot(dZ, W.T)
    dW = np.dot(A2.T, dZ)
    dB = np.dot(dZ.T, np.ones(N))
    
    dA = np.reshape(dA2, A.shape)
    return dA, dW, dB

def relu_forward(Z):
    """
    Inputs: Z
    Ouputs: A
            cache
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Inputs: dA
            cache
    Outputs: dA
    """
    Z = cache
    dA = np.array(dA, copy=True)
    dA[Z <=0] = 0

    return dA

def cross_entropy(F, y):
    """
    Inputs: F
            y
    Outputs: loss
             dF
    """

    n = F.shape[0]
    yint = y.astype(int)

    probability = np.exp(F-np.max(F, axis=1, keepdims=True))
    probability /= np.sum(probability, axis=1, keepdims=True)
    loss = -np.sum(np.log(probability[np.arange(n), yint]))/n

    dF = probability.copy()
    dF[np.arange(n), yint] -= 1
    dF /= n

    return loss, dF
