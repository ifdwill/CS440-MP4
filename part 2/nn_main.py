from neural_network import minibatch_gd, test_nn
import numpy as np
import time
import matplotlib.pyplot as plt

def init_weights(d, dp):
    return 0.01 * np.random.uniform(0.0, 1.0, (d, dp)), np.zeros(dp)

def plot_visualization_layer1(images, cmap):
    """Plot the visualizations 
    """    
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax[i%2, i//2].imshow(images[:, i].reshape((28, 28)), cmap=cmap)
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
    plt.show()

def plot_visualization(images, cmap):
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax[i%2, i//2].imshow(images[:, i].reshape((16, 16)), cmap=cmap)
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
    plt.show()

if __name__ == '__main__':
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    load_weights = False #set to True if you want to use saved weights

    if load_weights:
        w1 = np.load('w1.npy')
        w2 = np.load('w2.npy')
        w3 = np.load('w3.npy')
        w4 = np.load('w4.npy')

        b1 = np.load('b1.npy')
        b2 = np.load('b2.npy')
        b3 = np.load('b3.npy')
        b4 = np.load('b4.npy')
    else:
        w1, b1 = init_weights(784, 256)
        w2, b2 = init_weights(256, 256)
        w3, b3 = init_weights(256, 256)
        w4, b4 = init_weights(256, 10)

    start_time = time.time()
    num_epochs = 10
    #num_epochs = 30
    #num_epochs = 50

    w1, w2, w3, w4, b1, b2, b3, b4, losses = minibatch_gd(num_epochs, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, 10)
    print("---%s epochs: %s seconds ---" % (num_epochs, (time.time() - start_time)))
    np.save('w1', w1)
    np.save('w2', w2)
    np.save('w3', w3)
    np.save('w4', w4)

    np.save('b1', b1)
    np.save('b2', b2)
    np.save('b3', b3)
    np.save('b4', b4)
    #plt.plot(np.arange(num_epochs), losses)
    #plt.show()
    #wb1 = w1.copy()
    #wb1 = np.reshape(wb1, (784, 256))
    #plot_visualization_layer1(wb1, None)

    #wb2 = w2.copy()
    #wb2 = np.reshape(wb2, (256, 256))
    #plot_visualization(wb2, None)
    
    #wb3 = w3.copy()
    #wb3 = np.reshape(wb3, (256, 256))
    #plot_visualization(wb3, None)
    

    #wb4 = w4.copy()
    #wb4 = np.reshape(wb4, (256, 10))
    #plot_visualization(wb4, None)

    avg_class_rate, class_rate_per_class = test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, 10)
    print(avg_class_rate, class_rate_per_class)
