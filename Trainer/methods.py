import torch
import numpy as np  
from .base import Pinns
import pandas as pd
import csv
import time


def TemporalPINN(n_int=10000, n_sb=100, nt=10, delta_t=0.01, epochs=1000, device = "cpu", seed=42, save_path = './results/ADAM_test_1.pt', 
            pre_model_save_path = None, optimizer = "adam", lr = 1e-3, iters =1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    networks=[] # List to store networks for each time interval
    time_intervals = [] # List to store corresponding time intervals
    u_previous = torch.tensor(pd.read_csv('initial_0.csv').values[0:100,:], dtype=torch.float32, device=device) # Initial condition


    for i in range(1, nt+1):
        time_intervals.append(i*delta_t)

        Temporal_PINN = Pinns(n_int, n_sb, save_path, pre_model_save_path, device, delta_t, u_previous, optimizer)
            
        print(f"Training for time interval {i}")
        _, u_end = Temporal_PINN.fit(num_epochs=epochs, max_iter=iters, lr = lr, verbose=False)  # Fit the PINN

        u_previous = u_end.detach() # Update the initial condition for the next time interval
        networks.append(Temporal_PINN)
        print(f"Training for time interval {i} complete")

        # Save u_previous to i_output.csv
        output_filename = f"output_csv/{i+1}_output.csv"
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(u_previous.detach().cpu().numpy())




        

    








    