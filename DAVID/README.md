# Softwareproject by David Daniel Rosas Marquez

Hello and welcome to my part of the Softwareproject. In the Following Text I'll give a brief overview of my work throughout the semester, that I did for the course "Softwareprojekt: Machinelles Learnen für Lebenswissenschaftliche Daten". But first a couple of notes:

---

## Notes

1) This folder doesn't contain all of the scripts I wrote throughout the semester, just the ones I thought are the most relevant to the course and the "final attempts" for each project. I'll also always specify the files that need to be run and in which order, and if specific folder structures are needed to be able to reproduce the intended output.

2) A lot of the python scripts are extensively commented (especially for the first couple projcets). I chose to leave the comments in, because they were part of my process to try and better understand certain machine learning concepts (As I was completely new to the topic), and to rationalize what was happening in the code. But this also means that a lot of the comments might contain unclear or outright wrong statements, a lot of typos, grammar errors, redundancy and so on. This goes especially for comments where I am trying to understand/explain certain concepts.

3) The naming conventions for classes and variables and the coding style can be somewhat inconsistent. The code structure for the projects is also somewhat messy and not necessarily optimaly modular. I hope this doesn't cause too much confusion.

4) I divided my work into Projects called "Project_x" that thematically fit together. Some of them took multiple attempts or multiple weeks, but I decided to only upload the final working versions. I did this to provide an easier to follow overview of my work thoughout this semester.

---

With this in mind now follows the Overview.

---

# PROJECT 0

The First couple of weeks for me mostly involved getting into the topic of machine learning and artificial neural networks. For this we first implemented a simple linear regression using gradient descent, a technique also used for learning in neural networks. In the script "linear_regression.py" (runs on it's own) the idea is to interpolate some data points that follow a simple linear function with noise. Some diagrams for the loss surface are also plotted, to better be able to visualize the effects of gradient descent. 
The other project is in the script "digit_recog.py" (runs on it's own), which contains code for a simple multilayer perceptron trained to recognize digits from the mnist_784 dataset. This project got me to learn the basics of machine learning and artificial neural networks.
During the first couple of weeks we also had to do some reading on more complex NODE models, for which I chose the ACE_NODE. In "ACE_NODE.odp" you can see the slides for the presentation I made on the topic.

**difficulties:**  
EVERYTHING. Especially trying to build the sufficient understanding to begin to tackle the concepts inside the paper "ACE-NODE: Attentive Co-Evolving Neural Ordinary Differential
Equations"

**learned:**  
- What is a neural network
- Forward Pass and Backpropagation (superficially)  
- how do neural networks "learn" using gradient descent  
- How to code and train a basic neural network
- Basics of classification tasks

---

# PROJECT 1
By this point I had already developed a basic understanding of some concepts in machine learning and ANNs. The next project consisted in starting our journey through the world of Neural Ordinary Differential Equations, a machine learning paradigm consisting of modeling the continous dynamics of a systems state. To this end I coded minimal version of a NODE in the script "minimal_node.py" (runs on it's own) and tested it's interpolation capabilities on a synthetic dataset of datapoints following a sinus curve. The idea here was to implement it without the use of dedicated libraries (other than jax) and to create our own numerical integration function using Euler's method. This attempt was sadly not very succesful and the model fails to interpolate the sinus curve, even without noise (probably due to some bug). The next week our task was to implement the same minimal architecture but with the use of the dedicated libraries equinox (for neural network), diffrax (for the integrator) and optax (for the optimizer). We tested the model on a dataset of covid infections over time for interpolation, and on a dataset of lynx and hare populations over time for interpolation and extrapolation. I also tested the model on the same synthetic dataset as before but with vastly better results. All in all the greatest accomplishment of my model was it's ability to capture the periodic behaviour of the populations for some runs of the extrapolation task. It also performed okay for the interpolation of covid cases. This can be tested out with the "min_node.py" script by choosing the experiment in the main function. It requires the "covid_data.npy" and "LH_data.npy" files to run.

**difficulties:**
The difficulties for this task were mainly to understand the differences of this new ML paradigm to the previous discrete model I had coded to recognize digits. Getting familiar with the Python libraries was also a big part of the work during these weeks. Another difficulty I had was to feed the node model with the lynx and hare population data. Up to this point it was not intuitive to me that time series could consist of multiple features. So thanks to the lynx and hare dataset I became more familiar with the concept of multivariate time series and how to feed the model with this data correctly, so that it could learn the interactions between the different variables.

**learned:**
- How to implement a minimal Node Architecture and a simple integrator using Euler's method
- The basics of the equinox, diffrax and optax libraries
- The importance of Normalizing the data and tweaking the Hyperparameters of a model to achieve better results
- What a multivariate time series is (through lynx and hare experiment)

---

# PROJECT 2

**difficulties:**

**learned:**

---

# PROJECT 3

**difficulties:**

**learned:**

---

# PROJECT 4

**difficulties:**

**learned:**

---

# PROJECT 5

**difficulties:**

**learned:**
