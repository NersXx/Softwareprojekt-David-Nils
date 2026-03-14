# WE ARE UUSING GRADIENT DESCENT HERE TO SOLVE THE LINEAR REGRESSION (ITERATIVE STEPS) EVEN THOGH CAN BE SOLVED ANALITICALLY (SO THERE IS AN EXACT SOLUTION)
# THE REASON FOR THIS IS MAINLY THAT IS THE TIME COMPLEXITY FOR THE NORMAL SOLUTION; WHICH INVOLVES CALCULATING THE INVERSE OF A MATRIX (O(n^3)) SO GRADIENT
# DESCENT IS MUCH FASTER AND PREFERRABLE FOR VERY LARGE DATASETS!


"""How partial differnetiation works and why we use it (google ai):

A partial derivative measures how a multivariable function changes when you change just one of its variables, holding all others constant. It's found by 
treating all variables except the one of interest as constants and then taking the derivative with respect to that single variable. 
For a function like f(x,y), the partial derivative with respect to x is written as ∂f/∂x or f_{x} 
(∂ is the symbol for partial derivative)

How it works     
    Isolate a variable: Choose one variable to focus on (e.g., \(x\)).
    Treat others as constants: Mentally treat all other variables (e.g., y, z) as if they were numbers that do not change.  (herin lies the key)
    Differentiate: Perform the standard differentiation rules on the function with respect to the chosen variable!

Example: For f(x,y) = x^2y + 3xy^2:
    ∂f/∂x = 2xy + 3y^2
    (you can also write it as ∂/∂x * (x^2y + 3xy^2) <- as this is f in this case)
    
    ∂f/∂y = x^2 + 6xy
"""


import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import time


def gradient_descent(params, x, y, history):
    for i in range(200):
        grads = grad_fn(params, x, y)
        #zip(params, grads) is basically creating an output ((param1, grad1), (param2, grad2))
        params = [p - learning_rate * g for p, g in zip(params, grads)]
        history.append(params)
        if i % 20 == 0:
            #:3d tells us to print in decimal format (as oposed to binary) with 3 what?
            print(f"Step {i:3d}: m = {params[0]:.3f}, b = {params[1]:.3f}, loss = {loss_fn(params, x, y):.4f}")
    print(f"\nLearned parameters: m = {params[0]:.3f}, b= {params[1]:.3f}")


# Model prediction
def predict(params, x):
    m, b = params
    return m * x + b    #slope m and intercept b


#we define a loss function to determine how much loss our prediction generates and how to make it better
#Here specifically we are calculating the mean squared error MSE. Meassures how far off our predictions are
#from the true data points
def loss_fn(params, x, y):
    y_pred = predict(params, x)
    return jnp.mean((y - y_pred) ** 2)  #we return the mean squared error

#Our trainging goal is then to minimize the loss!



def plot_graph(x, y):
    plt.figure()
    plt.scatter(x, y, color ='blue', alpha=0.6)
    plt.title("Synthetic Data: y = 3x + noise")
    plt.xlabel("x")
    plt.ylabel("y")

#Normalizing the data!!!  in this case we use ZERO MEAN AND UNIT VARIANCE: x′= (​x−μx​​)/σx and y′= (​y−μx​​)/σy        where σ is the std error and μ the mean!
# https://www.geeksforgeeks.org/data-science/what-is-zero-mean-and-unit-variance-normalization/
def normalize(data):
    data = (data - jnp.mean(data))/ jnp.std(data)
    return data

def undo_normalize(data):
    data = data
    return data


# Random key
key = random.key(int(time.time()))
keys = random.split(key, 7)

data_points = 15
noise_factor = 50

#uniformly sampled
x = jnp.linspace(0,100,data_points) #creates an array of data_points elements with evenly spaced values from start to stop included
noise = random.normal(keys[0], shape = x.shape) * noise_factor #normal is for normalverteilung (gaussian distribution)
#x.shape means (the output of normal is an array) we want an array with the same shape as x, so with the same amount of values.
#so we are generating len(x) amount fo random values from a normal distribution with mean = 0 and standard deciation = 1 as default!
y = 3 * x + noise #we generate the dependant variable y, so an array with len(x) by applying the function to x (with added noise)

#irregular sampling
x_irr = random.uniform(keys[1], shape = (data_points,),minval = 0, maxval = 100) #tells python make a 1D array with data_points elements, comma is required to tell python it is a tuple with one element!
#shape expects us to describe ALL array dimensions! in this case its one dimensional, but it could be (depth, rows, cols), etc (so shape tells jax the array dimensions to generate)
#x_irr = jnp.sort(x)    #why are we doing this?
jnp.sort(x_irr) # because we want to have the random x intervals in order!


noise_irr = random.normal(keys[2], shape = x_irr.shape) * noise_factor
y_irr = 3 * x_irr + noise_irr

"""
#for uniform data
x = normalize(x)
y = normalize(y)

#for irregular data
x_irr = normalize(x_irr)
y_irr = normalize(y_irr)
"""

#we initialize the parameters randomly
params = jnp.array([random.normal(keys[3]), random.normal(keys[4])])
params_irr = jnp.array([random.normal(keys[5]), random.normal(keys[6])])

#compute gradient fucntion from the loss function!
#each component we get here is a partial derivative! One for how the loss changes when the slope changes
#and the other for how the loss changes when the intercept changes!: ∇L=(∂L/∂m​, ∂L/∂b​)
grad_fn = jax.grad(loss_fn)

#grad_fn(params, x, y) returns a tuple of the partial derivate swith respect to its first argument (params) 
#so one partial derivate for each of the two params, everything else is kept constant. Then Jax evaluates the
#function given our real parameters (real values), so what the gradient of the loss_function is, when we evaluate 
#for our specific slope (m) or our specific intercept (b). (Because remember: L(m,b)=(y−(mx+b)))^2, is a function 
#that calculates the loss depending on m and b, our slope and intercept. And we want ∇L(m,b), which gives us the 
#gradient (betrag des gradients ist die steigung, aslo gradient zeigt richtung und die größe des anstiegs!) functions
#for both The loss dependent on the slope and the intercept. Then for our specific value of m and b, we can calculate
#the gradient at that point.

#------------------------------------------------------------------------------------------------------------------------------------------------------------

#how to choose the learning rate correctly? There are different methods, but something to always keep in mind is if normalizing your data is important, because
#sometimes the gradient magnitudes might be extremely large and cause a lot of unbalance. So shrinking thease magnitudes makes gradient descent easier. 
#we choose a learning rate η, which is a small positive number that controls the step size of the learning algorithm!
#mathematically:
#   mnew = mold - η ∂L/∂m
#   bnew = bold - η ∂L/∂b
#       gradient tells us which way the loss increases the fastest, so we move in the opposite direction
#   basically if at our current m, the slope is positive, it means going further right (increasing the value of m) will increase the loss, so we move left (decrease m)
#   if at our current param, the slope is negative, it means going further right (increasing the value of p) will decrease the loss so we move right (increment p)
# If slope > 0 → subtracting moves left → lowers loss
# If slope < 0 → subtracting moves right → lowers loss
#the gradient ∇L always points uphill (the steepest ascent) ! we go in the opposite driection, downhill

learning_rate = 0.00005
history = []
history_irr = []


#Gradient descent loop 1
gradient_descent(params, x, y, history)   

#gradient descent loop 2 (irr)
gradient_descent(params_irr, x_irr, y_irr, history_irr)



#STEP 1 AND 2 VISUALIZING DATA POINTS AND PREDICTED LINE
def plot_training_progress(x, y, history):
    plt.figure(figsize = (7,5))
    plt.scatter(x, y, color="blue", label="data", alpha=0.6)
    
    #plot a few selected lines to show evolution
    #enumerate returns an index list, we take every fifth items
    for i, (m,b) in enumerate(history[::len(history)//5]):
        y_pred = m * x + b
        plt.plot(x, y_pred, label = f"step {i}")

    plt.plot(x, predict(history[-1], x), color="red", label="Final Fit", linewidth = 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Training Progress")
    plt.legend()
    plt.show()

plot_training_progress(x ,y ,history)

from mpl_toolkits.mplot3d import Axes3D

#3D plot off the loss surface!
def plot_loss_surface(x, y):
    #this would be our X Axis
    m_vals = jnp.linspace(-5, 10, 100)
    #this would be our Y Axis
    b_vals = jnp.linspace(-300, 300, 100) #maybe change this axis a bit
    M, B = jnp.meshgrid(m_vals, b_vals)
    
    #This would be our Z axis Showing the loss (height)
    #two dimensional list comprehension, python executes the inner list comprehension for each outer b
    Z = jnp.array([[loss_fn([m,b], x, y) for m in m_vals] for b in b_vals])
    
    #plot
    fig = plt.figure(figsize=(8,6)) #creates the window
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, B, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel("m (slope)")
    ax.set_ylabel("b (intercept)")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Surface")
    plt.show()
    
plot_loss_surface(x, y)

#vs x centered:

x_centered = x - jnp.mean(x)
y_centered = y - jnp.mean(y)
plot_loss_surface(x_centered, y_centered)


#Visualization of the partial derivatives!
#Remember the Vector is a Gradient of Partial Derivatives!!
# ∇L = [∂L/∂m, ∂L/∂b] each component tells us how fast and in which direction the loss changes when
#changing one parameter, while holding the other constant
#-> It is a 2D vector field in the (m,b)-plane !!! the direction -> direction of the steepest increase

#-> negative direction = the direction of the steepest decrease (used by gradient descent)
#-> jax returns to you the scalar value of each component of the vector for the current parameters (m,b)!
#-> the gradient vector field is computed by computing this vector for many points! not just one, (grad_fn() gives us the gradient vector at one point)
#the collection of these gives us our gradient field!

#2D gradient field (on m,b plane) arrows show the direction and magnitude of the steepest ascent

def gradient_field(x, y):
    m_vals = jnp.linspace(-2, 6, 20)
    b_vals = jnp.linspace(-50, 50, 20)
    
    M,B = jnp.meshgrid(m_vals, b_vals) #These are 2 paired 2D arrays, one a grid of all slopes and one a grid of all intercepts!
    #each row is a copy of m_vals (so the value stays the same throughout the row), each column is a copy of b_vals (the value stays the same throughout the column)
    
    dL_dm = jnp.zeros_like(M)   #returns an array of zeroes with the same shape
    dL_db = jnp.zeros_like(B)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            grads = grad_fn(jnp.array([M[i,j],B[i,j]]), x, y)  #we compute the gradient vector for each pari of values m and b
            # M[i,j] is jax and numpy syntax, it accesses element directly at coordinates (i,j) while
            # M[i][j] does the same but slower, as it first extracts row i and then element j of that row (Avoids creating an intermediate array)
            
            dL_dm = dL_dm.at[i, j].set(grads[0])
            dL_db = dL_db.at[i, j].set(grads[1])
    
    #we normalize the vectors
    U = -dL_dm
    V = -dL_db
    #N = jnp.sqrt(U**2 + V**2)
    U_norm = U / jnp.max(jnp.abs(U)) if jnp.max(jnp.abs(U)) > 0 else U
    V_norm = V / jnp.max(jnp.abs(V)) if jnp.max(jnp.abs(V)) > 0 else V
    #the scale of the slope component is much larger than that of the intercept! so proper normalization is needed for good visualization!
    
    Z = jnp.array([[loss_fn([m,b], x, y) for m in m_vals] for b in b_vals])#This is just for reference but it will plot wrong if we dont normalize this too
    plt.figure(figsize=(10,6))
    plt.contour(M, B, jnp.log10(Z), levels =20, cmap='viridis') #reuse Z from loss surface, we are using log10 to scale the magnitudes smaller and make them easier to visualize
    plt.quiver(M, B, U_norm, V_norm, color='red', angles='xy', scale_units = 'xy', scale =1) # negative gradients = descent direction
    plt.xlabel("m (slope)")
    plt.ylabel("b (intercept)")
    plt.title("Gradient Field of the Loss Function")
    plt.show()

gradient_field(x, y)
gradient_field(x_centered, y_centered)



#later learn to use batch vectorization!
#later learn how to animate the vector field!