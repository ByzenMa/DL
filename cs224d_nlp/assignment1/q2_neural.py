import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h_layer_i = np.dot(data, W1)
    h_layer_b = h_layer_i + b1
    h_layer_o = sigmoid(h_layer_b)
    o_layer_i = np.dot(h_layer_o, W2)
    o_layer_b = o_layer_i + b2
    o_layer_o = sigmoid(o_layer_b)
    sm = softmax(o_layer_o)
    cost_before_sum = -labels * np.log(sm)
    cost = np.sum(cost_before_sum.T, axis=0).T
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    do_layer_o = sm - labels
    do_layer_b = sigmoid_grad(o_layer_o) * do_layer_o
    do_layer_i = do_layer_b
    db2 = do_layer_b
    dh_layer_o = np.dot(do_layer_i, W2.T)
    dW2 = np.dot(h_layer_o.T, do_layer_i)
    dh_layer_b = sigmoid_grad(h_layer_o) * dh_layer_o
    dh_layer_i = dh_layer_b
    db1 = dh_layer_b
    dW1 = np.dot(data.T, dh_layer_i)
    gradW1 = dW1
    gradb1 = db1
    print(gradb1.shape)
    gradW2 = dW2
    gradb2 = db2
    # print(gradb2.shape)
    # print(gradW2.shape)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
                                                         dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
