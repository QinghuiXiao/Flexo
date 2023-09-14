import torch
import numpy as np  
from .base import Pinns
from .base import Pinns2
import pandas as pd
import csv
import time


def TemporalPINN(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    networks=[] # List to store networks for each time interval
    time_intervals = [] # List to store corresponding time intervals
    u_previous = torch.tensor(pd.read_csv('initial_0.csv').values[0:args.n_int, :], dtype=torch.float32, device=args.device) # Initial condition


    for i in range(1, args.nt+1):
        time_intervals.append(i*args.delta_t)

        Temporal_PINN = Pinns(config=args, u_previous_=u_previous)
            
        print(f"Training for time interval {i}")
        u_end = Temporal_PINN.fit(num_epochs=args.epochs, max_iter=args.iters, lr = args.lr, verbose=False)  # Fit the PINN

        u_previous = u_end # Update the initial condition for the next time interval
        networks.append(Temporal_PINN)
        print(f"Training for time interval {i} complete")

        # Save u_previous to i_output.csv
        output_filename = f"output_csv/{i}_output.csv"
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(u_previous.cpu().numpy())

def CtemporalPINN(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    Ctemporal_PINN = Pinns2(config=args)
    hist = Ctemporal_PINN.fit(num_epochs=args.epochs, max_iter=args.iters, lr=args.lr, verbose=True)
    Ctemporal_PINN.plotting()





        

    








    