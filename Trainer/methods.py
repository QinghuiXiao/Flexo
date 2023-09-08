import torch
import numpy as np  
from .base import Pinns
import pandas as pd
import csv
import time


def TemporalPINN(n_int=10000, n_sb=100, nt=10, delta_t=0.01, epochs=1000, device = "cpu", seed=42, save_path = './results/ADAM_test_1.pt', 
            pre_model_save_path = None, optimizer = "Adam", lr = 1e-3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    networks=[] # List to store networks for each time interval
    time_intervals = [] # List to store corresponding time intervals
    u_previous = torch.tensor(pd.read_csv('initial_0.csv').values, dtype=torch.float32, device=device) # Initial condition


    for i in range(1, nt+1):
        time_intervals.append(i*delta_t)

        Temporal_PINN = Pinns(n_int, n_sb, save_path, pre_model_save_path, device, delta_t, u_previous)

        if optimizer == "Adam":
            optimizer = torch.optim.Adam(Temporal_PINN.approximate_solution.parameters(),
                                lr=float(0.001))
        elif optimizer == "LBFGS":
            optimizer = torch.optim.LBFGS(Temporal_PINN.approximate_solution.parameters(),
                                  lr=float(0.5),
                                  max_iter=1000,
                                  max_eval=50000,
                                  history_size=150,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
            
        print(f"Training for time interval {i}")
        _, u_end = Temporal_PINN.fit(num_epochs=epochs, optimizer=optimizer, verbose=True)  # Fit the PINN

        u_previous = u_end # Update the initial condition for the next time interval
        networks.append(Temporal_PINN)
        print(f"Training for time interval {i} complete")

        # Save u_previous to i_output.csv
        output_filename = f"{i+1}_output.csv"
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(u_previous.detach().cpu().numpy())




        

    








    