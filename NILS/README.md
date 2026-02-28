# Softwareproject by Nils Enno Lenwerder

Hello and welcome to my part of the Softwareproject. In the Following Text I'll give a brief overview of my work throughout the semester, that I did for the course "Softwareprojekt: Machinelles Learnen für Lebenswissenschaftliche Daten". But first a couple of notes:

---

## Notes

1) The main folder doesn't contain all of the scripts I wrote throughout the semester, just the ones I thought are the most relevant to the course and the "final attempts" for each project. I'll also always specify the files that need to be run and in which order, and if specific folder structures are needed to be able to reproduce the intended output.

2) A lot of the python scripts are extensively commented. I chose to leave the comments in, because they were part of my process to try and better understand certain machine learning concepts (As I was completely new to the topic), and to rationalize what was happening in the code. Not all of the comments and the code was written by me, as I used LLMs to help me understand the topic and fix some of my bugs. But this also means that a lot of the comments might contain unclear or outright wrong statements, some typos, grammar errors, redundancy and so on. This goes especially for comments where I am trying to understand/explain certain concepts.

3) The naming conventions for classes and variables and the coding style can be somewhat inconsistent. The code structure for the projects is also somewhat messy and not necessarily optimaly modular. I hope this doesn't cause too much confusion.

4) I divided my work into Projects called "Project_x" that thematically fit together. Some of them took multiple attempts or multiple weeks, but I decided to only upload the final working versions. I just threw some code checkpoints in the folder "old". I did this to provide an easier to follow overview of my work thoughout this semester.

---

With this in mind now follows the Overview.

---

# PROJECT 0

The First couple of weeks for me mostly involved getting into the topic of machine learning and artificial neural networks. For this we first implemented a simple feedforward neural network to classify data from the Iris dataset. This part of the project helped in understanding the basics of neural networks, including forward propagation, backpropagation, and gradient descent. In the script "Project 0-2" notebook section 32 (execution number) (runs on it's own). The goal is to classify the Iris flowers into their respective categories based on their features.
During the first couple of weeks we also had to do some reading on more complex NODE models, for which I chose the ACE_NODE. In "ACE_NODE.odp" in Davids section you can see the slides for the presentation on the topic.

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
By this point I had already developed a basic understanding of some concepts in machine learning and ANNs. The next project consisted in starting our journey through the world of Neural Ordinary Differential Equations, a machine learning paradigm consisting of modeling the continous dynamics of a systems state. To this end I first implemented a MLP to interpolate the sin function. After that I tried to code a minimal version of a NODE in the notebook cell 49 (runs on it's own) and tested it's interpolation capabilities on a synthetic dataset of datapoints following a sinus curve.

**difficulties:**
Getting the NODE to work with the Sin curve. At the beginning I completely ignored the X Axis on the Sin curve, so i was alway getting a 0 predicted. Getting familiar with the Python libraries was also a big part of the work during these weeks. 
Getting the correct matrix shapes was also quite difficult, as I was still quite new to the concept of ML.

**learned:**
- The basics of the optax library
- The importance of Normalizing the data and tweaking the Hyperparameters of a model to achieve better results
- What a multivariate time series is (through lynx and hare experiment)

---

# PROJECT 2

The next week our task was to implement the models with the use of the dedicated libraries equinox (for neural network), diffrax (for the integrator) and optax (for the optimizer). We tested the model on a dataset of covid infections over time for interpolation, and on a dataset of lynx and hare populations over time for interpolation and extrapolation. All in all the greatest accomplishment of my model was it's ability to capture the interpolated graph for the Covid data. It also performed okay for the extrapolation task, but failed to capture the periodical behavior and the Hunter-Prey relationship. This can be tested out with the cell 6 for the Covid Data and cells 10, 27, 41, 53 for the Lynx and Hares. It requires the "covid_data.npy" and "LH_data.npy" files to run.

**difficulties:**
one again using the Python libraries was a somewhat big part of the work. Another difficulty I had was to use the multiple features of the LH Data. 
**learned:**
- The basics of the equinox, diffrax and optax libraries
- The importance of Normalizing the data and tweaking the Hyperparameters of a model to achieve better results
- What a multivariate time series is (through lynx and hare experiment)

---

# PROJECT 3

The following weeks, we were tasked with predicting the Latent variable alpha of different Spiral datasets, including missing data with varying spaces with missing data. In the first Week was some error with the Data, that was fixed in the second week. To accomplish this task I implemented a ODE-RNN, which updates every timestep where observations arrive. I also came across the idea to use masks, which I Implemented for the missing values, however I am not currently using them for the loss. In the folder Project_3_(Spirals_week2) are the final files. The File Spiral.py requires the spiral data. Cusom it uses the Spirals_25 dataset. The Notebook Spirals.ipynb requires the spirals data aswell and it contains some files from production. I am quite intersted in multi device training, so I tried to implement a multi GPU version of the code, because the Data size was quite large and my laptop took a while to compute.

**difficulties:**
- getting comfortable with the RNN part
- Shapemanagement 
- Diffrax integration with hidden state
- time normalizing and time handling
- pmap: replicate model, shard batches, average loss...
**learned:**
- ODE + RNN correlation
- shape discipline is essential
- core functions for multi Gpu computing
- Time: normalized, denormalize
---

# PROJECT 4

In Project 4 we shifted from toy interpolation tasks to a clinical classification problem: predicting whether a patient develops sepsis from multivariate time-series data. I first experimented with an attention-enhanced NODE backbone and then integrated it into a classification pipeline with a readout head. A major part of this project was engineering: building a robust NPZ-based data pipeline, handling variable sequence lengths, normalizing features, and setting up stable training/evaluation loops.
Because sepsis labels are imbalanced, I focused on loss design and metrics, using focal loss and tracking AUC, recall, F1, and confusion matrix values instead of relying on loss alone. I also experimented with multi-device training and stronger stabilization techniques (learning-rate warmup/decay, gradient clipping, dropout, careful initialization). This project was the transition from “model prototype” to a more complete ML workflow.
Because I went through many iterations during this phase of the software project, I only included the most relevant files in the folder "Project_4". Other intermediate versions are in "old/Model". I consider Project 4 to end at the point where David and I started working on the same model; after that, the continuation is in the "Final_project" folder. To run the Project 4 files, the created NPZ files are required (same data basis as in the final project). I also tried to transfer the attention-enhanced NODE approach to some of the older tasks, but the complexity was too high at that stage.

**difficulties:**
- training instability
- data preparation and building a reliable NPZ pipeline
- class imbalance in sepsis detection (many non-sepsis samples)
- tuning hyperparameters (learning rate, batch size, dropout, hidden size)
- making multi-device training work reliably (pmap, sharding, replicated states)
- handling variable sequence lengths, missing values and shape consistency

**learned:**
- How to move from a research prototype to a more robust training pipeline
- The importance of stable optimization for NODE-style models (clipping, scheduling, regularization)
- How strong data preprocessing and normalization affect model quality
- How to design a classification wrapper around a sequence model for clinical prediction
- How to evaluate classification models beyond loss (AUC, confusion matrix, recall, F1)
- Better practical understanding of JAX + Equinox + Optax in a larger end-to-end project

