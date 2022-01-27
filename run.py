import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# After loading the MNIST data, we'll set up a Network with 30 hidden neurons. 
net = network.Network([784, 30, 10])

# Finally, we'll use stochastic gradient descent to learn from the MNIST 
# training_data over 30 epochs, with a mini-batch size of 10, and a learning rate of ni=3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


