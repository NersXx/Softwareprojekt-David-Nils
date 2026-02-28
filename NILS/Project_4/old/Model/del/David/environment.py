#Group: David, Nils

import sys
import time

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


import spiral
import ACE_ODE_RNNv2 as ace_ornn
import norm


def main() -> int:
    
    key = random.key(int(time.time()))
    
    choice = 0
    while choice == 0:
        try:
            user_input = input("please insert a number for the experiment to be executed:\n"
                    "1: Full trajectory (100 time steps)\n"
                    "2: 75 randomly selected data points\n"
                    "3: 50 randomly selected data points\n"
                    "4: 25 randomly selected data points\n")
            choice = int(user_input)
        except Exception as e:
            print(e,"\n")
            print("Please Enter a valid choice between 1 and 4")
         
    #this is temporary, but make a loop that runs through experiments with different parameters every time (well right now they all use the same params) 
    alpha_pred = {} 
    for choice in [100, 75, 50, 25]:   
        match choice:
            case 100:
                alpha_pred[str(choice)] = run_experiment(dataset = spiral.spirals_100, 
                            hidden_dim = 32, solver_width = 32, output_nn_width = 32,
                            solver_depth= 2, output_nn_depth= 2, 
                            epochs = 10, learning_rate = 1e-3, batch_size= 50,
                            X_normalizer= norm.ZScoreNorm(),
                            y_normalizer= norm.ZScoreNorm(),
                            ts_normalizer= norm.MinMaxNorm(), 
                            key = key,
                            file_name = "pred_spirals_100.npy")
            case 75:
                alpha_pred[str(choice)] = run_experiment(dataset = spiral.spirals_75, 
                            hidden_dim = 32, solver_width = 32, output_nn_width = 32,
                            solver_depth = 2, output_nn_depth = 2,
                            epochs = 100, learning_rate = 1e-3, batch_size = 50,
                            X_normalizer = norm.ZScoreNorm(),
                            y_normalizer = norm.ZScoreNorm(),
                            ts_normalizer = norm.MinMaxNorm(),
                            key = key,
                            file_name = "pred_spirals_75.npy")
            case 50:
                alpha_pred[str(choice)] = run_experiment(dataset = spiral.spirals_50, 
                            hidden_dim = 32, solver_width = 32, output_nn_width = 32,
                            solver_depth = 2, output_nn_depth = 2,
                            epochs = 100, learning_rate = 1e-3, batch_size = 50,
                            X_normalizer = norm.ZScoreNorm(),
                            y_normalizer = norm.ZScoreNorm(),
                            ts_normalizer = norm.MinMaxNorm(),
                            key = key,
                            file_name = "pred_spirals_50.npy")
            case 25:
                alpha_pred[str(choice)] = run_experiment(dataset = spiral.spirals_25, 
                            hidden_dim = 32, solver_width = 32, output_nn_width = 32,
                            solver_depth = 2, output_nn_depth = 2,
                            epochs = 100, learning_rate = 1e-3, batch_size = 50,
                            X_normalizer = norm.ZScoreNorm(),
                            y_normalizer = norm.ZScoreNorm(),
                            ts_normalizer = norm.MinMaxNorm(),
                            key = key,
                            file_name = "pred_spirals_25.npy")
            
    np.savez('david_nils.npz', **alpha_pred)    
    
       
    return 0





def run_experiment(*, dataset: spiral.SpiralData, 
                   hidden_dim, solver_width, output_nn_width, solver_depth, output_nn_depth, 
                   epochs, learning_rate, batch_size, 
                   X_normalizer :norm.Normalizer , y_normalizer: norm.Normalizer, ts_normalizer: norm.Normalizer,
                   key, file_name):
      
    #**args could be used later to allow for extra keywords (and this function can latter decide if to handle them or not!)
    
    model_key, train_key = random.split(key)
    data_train, alpha_train, data_validation, alpha_validation, data_test = dataset.extract_data()
    #data can be randomly sampled
        
    #Creating model
    
    input_dim = data_train.shape[2] - 1     #only the trajectories x and y without the time step t (that one is used for the rnn step and the ode solver)
    output_dim = alpha_train.shape[1]
    model = ace_ornn.ACE_ODE_RNN(input_dim, output_dim, hidden_dim, solver_width, output_nn_width,
                            solver_depth, output_nn_depth, key = model_key)
    
    
    #Normalizing Data --------------------------------------------------
    X_train, ts_train = unpack(data_train) 
  
    X_normalizer.init(X_train, axis = (0,1))
    y_normalizer.init(alpha_train, axis = (0,1))  
    ts_normalizer.init(ts_train, axis = None)

    X_train_norm = X_normalizer(X_train)
    alpha_train_norm = y_normalizer(alpha_train)
    ts_train_norm = ts_normalizer(ts_train)
    
    #Training Model
    model = ace_ornn.training_loop(X_train_norm, alpha_train_norm, ts_train_norm,
                                  model, epochs, learning_rate, batch_size, key = train_key)
    
    #------------------
    #VALIDATION DATA
    #------------------
    
    #Validation
    print("\nValidating model on Validation data...\n")
    X_val, ts_val = unpack(data_validation) 
    X_val_norm, ts_val_norm = X_normalizer(X_val), ts_normalizer(ts_val)
    
    alpha_val_pred_norm = model.batched_call(ts_val_norm, X_val_norm)
    alpha_val_pred = y_normalizer(alpha_val_pred_norm, denormalize = True)
    
    mse_validation = jnp.mean((alpha_validation - alpha_val_pred)**2)
    print(alpha_val_pred.shape)
    
    #Plots
    plot_validation_overview(alpha_validation, alpha_val_pred, mse_validation)
    
    #-----------
    #TEST DATA
    #-----------
    
    #prediction on test data
    print("\nCreating prediction from test_data...\n")
    X_test, ts_test = unpack(data_test)
    X_test_norm, ts_test_norm = X_normalizer(X_test), ts_normalizer(ts_test)
    
    alpha_test_pred_norm = model.batched_call(ts_test_norm, X_test_norm)
    alpha_test_pred = y_normalizer(alpha_test_pred_norm, denormalize = True)
    
    print(alpha_test_pred.shape)
    
    #plot 
    plot_test_overview(X_test, alpha_test_pred)
    
    
    #create file #--------------------------------------------------------------------
    #np.save(file_name, alpha_test_pred)
    
    return alpha_test_pred
    

 
 
 
    
def unpack(data):
    """function to preprocess the data, maybe make it more extensible in the future,
    allowing for specifying slice and axis etc"""
    #X_train_ts has shape (Batch, N, 3)
    ts = data[:,:,0:1]     #has shape (Batch, N, 1)
    ts = ts[..., 0]        #has shape (Batch, N)
    X =  data[:,:,1:3]     #has shape (Batch, N, 2)
    return X, ts








#---------------
# PLOT FUNCTIONS
#---------------

def plot_validation_overview(alpha_true, alpha_pred, mse):
    # Flatten everything to 1D
    alpha_true = jnp.array(alpha_true).flatten()
    alpha_pred = jnp.array(alpha_pred).flatten()
    residuals = alpha_pred - alpha_true
    abs_error = jnp.abs(residuals)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Validation Overview (MSE = {mse:.3e})", fontsize=16)

    # ---------------------------------------
    # 1) Predicted vs True α
    # ---------------------------------------
    ax = axs[0, 0]
    ax.scatter(alpha_true, alpha_pred, s=10, alpha=0.6)
    ax.plot([alpha_true.min(), alpha_true.max()],
            [alpha_true.min(), alpha_true.max()],
            'r--', label="Perfect Prediction")
    ax.set_xlabel("True α")
    ax.set_ylabel("Predicted α")
    ax.set_title("Predicted vs True α")
    ax.grid(True)
    ax.legend()

    # ---------------------------------------
    # 2) Histogram of residuals
    # ---------------------------------------
    ax = axs[0, 1]
    ax.hist(residuals, bins=40, edgecolor='black', alpha=0.8)
    ax.set_xlabel("Residual (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(True)

    # ---------------------------------------
    # 3) Error vs Magnitude
    # ---------------------------------------
    ax = axs[1, 0]
    ax.scatter(alpha_true, abs_error, s=10, alpha=0.6)
    ax.set_xlabel("True α")
    ax.set_ylabel("|Prediction Error|")
    ax.set_title("Error vs Magnitude of α")
    ax.grid(True)

    # ---------------------------------------
    # 4) Distribution of α values
    # ---------------------------------------
    ax = axs[1, 1]
    ax.hist(alpha_true, bins=40, alpha=0.5, label="True α")
    ax.hist(alpha_pred, bins=40, alpha=0.5, label="Predicted α")
    ax.set_xlabel("α value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of α (True vs Predicted)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




def plot_test_overview(X_test, alpha_pred):
    alpha_pred = jnp.array(alpha_pred).flatten()
    n = len(alpha_pred)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Test Data Overview (Unsupervised Output Analysis)", fontsize=16)

    # ------------------------------------------------
    # 1) Histogram of predicted α
    # ------------------------------------------------
    ax = axs[0, 0]
    ax.hist(alpha_pred, bins=40, edgecolor='black', alpha=0.8)
    ax.set_xlabel("Predicted α")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted α")
    ax.grid(True)

    # ------------------------------------------------
    # 2) α density (smooth KDE)
    # ------------------------------------------------
    ax = axs[0, 1]
    try:
        kde = gaussian_kde(alpha_pred)
        xs = jnp.linspace(alpha_pred.min(), alpha_pred.max(), 300)
        ax.plot(xs, kde(xs))
    except Exception:
        # fallback if kde fails (e.g. constant data)
        ax.plot(alpha_pred, np.zeros_like(alpha_pred), '.')
    ax.set_xlabel("Predicted α")
    ax.set_ylabel("Density")
    ax.set_title("Density Estimate of Predictions")
    ax.grid(True)

    # ------------------------------------------------
    # 3) Example trajectory scatter, colored by α
    # ------------------------------------------------
    ax = axs[1, 0]
    idx = np.random.randint(0, n)      # pick random trajectory
    traj = jnp.array(X_test[idx])       # shape (T, 2)
    ax.scatter(traj[:, 0], traj[:, 1], s=20,
               c=np.linspace(0, 1, traj.shape[0]),
               cmap="viridis")
    ax.set_title(f"Example Trajectory (index={idx})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # ------------------------------------------------
    # 4) Predicted α vs test sample index
    # ------------------------------------------------
    ax = axs[1, 1]
    ax.plot(alpha_pred, '.', alpha=0.7)
    ax.set_xlabel("Trajectory Index")
    ax.set_ylabel("Predicted α")
    ax.set_title("Predicted α Across Test Set")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()






if __name__ == "__main__":
    sys.exit(main())