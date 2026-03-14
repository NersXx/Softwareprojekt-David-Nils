"""Random bit of maths:
Matrices can be used to represent linear transformations (functions
that map vectors to other vectors) matrices are fundamental to understanding
operations on vectors. Or they can be just arrays of numbers, they are in a
sense, just tools.
"""

import jax
import jax.numpy as jnp
import time
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml # We are gonna use the MNIST dataset to train the digit recognition!, we used the flattened version (in one vector) of sklearn
from sklearn.preprocessing import OneHotEncoder
from jax import random



def main() -> int:
    #fetch dataset from Open ML
    mnist = fetch_openml('mnist_784', version = 1, as_frame = False) #fetch the data as numpy arrays
    
    #2. Extract data and labels
    X, y = mnist.data, mnist.target
    
    #3. Convert labels from strings to integers
    y = y.astype(int)
    
    #4. SPlit into training (60k) and testing (10k) data
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    #5. Normalize values to [0, 1] Range (because each dimension of the vector contains a Value from 0 to 255)
    X_train = X_train/ 255.0
    X_test = X_test / 255.0
    
    #6. Convert to JAX arrays to be able to work with
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    
    #7. Visualize some of the digits
    fig, axes = plt.subplots(1, 5, figsize =(10, 3))
    for i, ax in enumerate(axes) :
        image = X_train[i].reshape(28, 28) #we turn the vector back to a matrix
        label = int(y_train[i])
        ax.imshow(image, cmap = 'grey')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.show()
    
    #8. Encode the output into a vector with dim = 10
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    
    random_key = random.key(int(time.time()))
    input_dim = X_train.shape[1]
    hidden_dim1 = 16
    hidden_dim2 = 16
    output_dim = y_train_encoded.shape[1]
    learning_rate = 0.01
    batch_size = 16
    epochs = 200

    params = init_params(input_dim, hidden_dim1, hidden_dim2, output_dim, random_key)
    params = training_loop(params, X_train, y_train_encoded, X_test, y_test_encoded)
    
    #Visualization
    while 1:
        try:
            user_input = int(input("Please input a number between 0 and 10000:\n"))
        except Exception as e:
            print(f"Unexpected Error: {e}/n")
            return 1

        predict(params, X_test, y_test_encoded, user_input)
    
    return 0

#later we could try to use *args to have a variable amount of hidden dimensions!
def init_params(input_dim, hidden_dim1, hidden_dim2, output_dim, random_key):
    #split random key for reproducability
    k1, k2, k3 = random.split(random_key, 3)
    
    #dictionary contaning all our weights and biases
    params = {"W1": random.normal(k1, (input_dim, hidden_dim1)) *jnp.sqrt(1./input_dim), #weights are drrawn from normal distribution and scaled down by √(1/n_in), a common heuristic for stable gradients.
              "b1": jnp.zeros((hidden_dim1,)),
              "W2": random.normal(k2, (hidden_dim1, hidden_dim2)) * jnp.sqrt(1./hidden_dim1),
              "b2": jnp.zeros((hidden_dim2,)),
              "W3": random.normal(k3,(hidden_dim2, output_dim)) *jnp.sqrt(1./hidden_dim2),
              "b3": jnp.zeros((output_dim,)),
              }
    return params

#The forward pass is the process where data flows forward through the neural network, from the input layer to the output, to make predictions!
#In mathematical terms the neural network is just a big function: y_{hat} ​=f (x;θ) where:
#   x = input data (the pixels values of a digit)
#   θ = all the network's parameters (weights and biases)
#   y_{hat} = predicted output(e.g., probabilities for digits 0-9)! (in a vector, and we choose the dimension with the highest value)

def forward(params, X):
    h1 = jax.nn.relu(jnp.dot(X, params['W1']) + params['b1'])    #we compute the dot product of all "pixels" (dim of our input vector), with all the weights "connections to one neuron" fro all neurons and get a vector of dim(hidden_layer1)
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])  #we use nonlinear activation with relu to learn complex patterns
    logits = jnp.dot(h2, params['W3']) + params['b3'] #this layer needs to produce raw scores? These are then passed to a softmax function when computing loss
    #If i forced all outputs to be nonnegative it would mess with the training?
    return logits


#y is our training data, large weights make the model more sensitive to small input changes, and we don't want that?
def loss_fn(params, X, y, l2_reg = 0.0001):
    
    logits = forward(params, X)
    probs = jax.nn.softmax(logits)  #we turn the unbounded numbers into probabilities between o and 1, that sum up to 1!
    #compute cross-entropy loss (https://en.wikipedia.org/wiki/Cross-entropy)
    
    #this is the cross-entropy loss, and then we add the l2_loss
    #probs is the predicted probability distribution, y our vectorized training data, we add 1e-8 to avoid taking log(0)
    #we take the mean across all samples in the batch, negative sign means large probabilities for the correct class lower the loss
    cross_entropy_loss = -jnp.mean(jnp.sum(y*jnp.log(probs + 1e-8), axis = 1)) #elementwise multiplies yields a vector of length N, then calculates the mean across the samples
    #we y*jnp.log(probs) gives us a matrix of shape [0,0,log(p_correct),0,..] and N samples of this shape: probs[i,j] models the predicted probability that sample i is class j
    ##the sum across axis 1, summs across the classes for each sample (row) -> gives us one number per sample, colapses the row into one number
    #so we have something of shape [log(0.5), log(0.1), log(0.7), ...] A vector with the predicted probability for each sample for each class
    #so basically how good was the model at actually predicting the correct class!
    #we then compute the negative averege across all of the probabilities, so what was the average loss (average penalty per sample)
    #-> negative signe because log(probability) is negative (probabilities < 1) so negating it will give us a positive number
    
    l2 = 0.0 #with this we try to prevent the weights from exploding
    #l2 regularization or weight decay keeps weights small by adding a penalty proportional to their squared magnitude    
    #biases only shift activation threshholds, they dont cause overfitting, we are doing elementwise squaring
    """
    for k in params:    #where params is a dict
        if k.startswith('W'):
            l2 += jnp.sum(params[k] ** 2) #apparently we use squares because it is differentiable
    l2_loss = l2_reg * l2
    """
    #using generator comprehension:
    l2_loss = l2_reg * sum(jnp.sum(params[k]**2) for k in params if k.startswith('W'))
    
    return cross_entropy_loss + l2_loss #penalty for large weights, this number is not bound!

@jax.jit
def train_step(params, X, y, lr):
    grads = jax.grad(loss_fn)(params, X, y)
    #jax preserves structure, so if params is a nested dict, grads will have the same shape and keys!
    #using dictionary comprehension
    return {k: params[k] - lr * grads[k] for k in params}
    # basically we return a dictionary, where the value stored at every key, Is the matrix 
    # containing the evaluated gradients for all specific weights for the partial derivative 
    # corresponding to that specific weight (The connection between the input layer and the neuron)
    # and this across all matrices

def accuracy(params, X, y):
    preds = jnp.argmax(forward(params, X), axis = 1) #computes logits, then finds index for largest value across each row, collapses it into a single vector of shape (N,) (predicted class per sample)
    targets = jnp.argmax(y, axis = 1) # gives out the true class index for each sample
    return jnp.mean(preds==targets) #preds == targets creates a boolean array of shape (N,) True if prediction matches target, false otherwise. We compute the avarage success across all predictions

def data_loader(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
#We usually slice the data set into mini batches because it increases efficiency, we don't have to load all at once into the cpu
#This aids to faster convergence, because it introduces some noise in the gradient?? -> helps avoid shallow local minima
#Better for hardware utilization -> vectorized operations on batches are faster than looping over single samples!

def training_loop(params, X_train, y_train, X_test, y_test, learning_rate = 0.01, batch_size = 16, epochs = 200):
    
    for epoch in range(epochs):
        for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
            params = train_step(params, X_batch, y_batch, learning_rate)
    
        train_acc = test_acc = 0
        if epoch % 10 == 0:
            train_acc = accuracy(params, X_train, y_train)
            test_acc = accuracy(params, X_test, y_test)
            print(f"Epoch {epoch}: Train acc ({train_acc:.4f}), Test acc ({test_acc:.4f})")
        
    print(f"Final Test Accuracy: {accuracy(params, X_test, y_test):.4f}")
    return params

def predict(params, X, y, index):
    # Get the sample and its label
    sample = X[index]
    label = y[index]
    
    #reshape sample into image
    image = sample.reshape(28, 28)
    
    #Make prediction using the trained neural network
    logits = forward(params, sample[None, :]) #We add a batch dimension
    probs = jax.nn.softmax(logits)
    predicted_class = jnp.argmax(probs)
    
    #plot the digit
    plt.imshow(image, cmap = 'grey')
    plt.title(f"Predicted: {predicted_class}, True: {jnp.argmax(label)}")
    plt.show()
    
    return predicted_class, jnp.argmax(label)


if __name__ =="__main__":
    sys.exit(main())