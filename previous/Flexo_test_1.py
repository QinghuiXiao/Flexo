import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Trainer.model import NeuralNet
import csv
import time

# import time
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


class Pinns:
    def __init__(self, n_int_, n_sb_, save_dir_, pre_model_save_path_, device_, delta_t_, u_previous_):
        self.pre_model_save_path = pre_model_save_path_
        self.save_dir = save_dir_

        self.n_int = n_int_
        self.n_sb = n_sb_
        self.delta_t = delta_t_

        self.device = device_
        self.u_previous = u_previous_

        self.domain_extrema = torch.tensor([[0, 50],  # x dimension
                                            [0, 50]])  # y dimension

        # Number of space dimensions
        self.space_dimensions = 2

        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=3,
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42).to(self.device)

        '''self.approximate_solution = MultiVariatePoly(3, 3)'''

        if pre_model_save_path_:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=2)

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[0, 0]
        xL = self.domain_extrema[0, 1]
        y0 = self.domain_extrema[1, 0]
        yL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 0] = x0
        input_sb_D = torch.clone(input_sb)
        input_sb_D[:, 1] = y0
        input_sb_R = torch.clone(input_sb)
        input_sb_R[:, 0] = xL
        input_sb_U = torch.clone(input_sb)
        input_sb_U[:, 1] = yL

        input_sb = torch.cat([input_sb_U, input_sb_D, input_sb_L, input_sb_R], 0)

        return input_sb

    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        return input_int

    def assemble_datasets(self):
        input_sb = self.add_spatial_boundary_points()  # S_sb
        input_int = self.add_interior_points()         # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb),
                                     batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)

        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int), batch_size=self.n_int,
                                      shuffle=False)

        return training_set_sb, training_set_int

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):

        P1_previous = self.u_previous[:, 0].reshape(-1, 1)
        P2_previous = self.u_previous[:, 1].reshape(-1, 1)
        P1_previous = P1_previous.to(self.device)
        P2_previous = P2_previous.to(self.device)

        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)
        varphi = u[:, 2].reshape(-1, 1)

        grad_varphi = torch.autograd.grad(varphi.sum(), input_int, create_graph=True)[0]
        varphi_1, varphi_2 = grad_varphi[:, 0], grad_varphi[:, 1]
        grad_varphi_1 = torch.autograd.grad(varphi_1.sum(), input_int, create_graph=True)[0]
        varphi_11 = grad_varphi_1[:, 0]
        grad_varphi_2 = torch.autograd.grad(varphi_2.sum(), input_int, create_graph=True)[0]
        varphi_22 = grad_varphi_2[:, 1]

        grad_P1 = torch.autograd.grad(P1.sum(), input_int, create_graph=True)[0]
        P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1]

        grad_P1_1 = torch.autograd.grad(P1_1.sum(), input_int, create_graph=True)[0]
        P1_11, P1_12 = grad_P1_1[:, 0], grad_P1_1[:, 1]
        grad_P1_2 = torch.autograd.grad(P1_2.sum(), input_int, create_graph=True)[0]
        P1_21, P1_22 = grad_P1_2[:, 0], grad_P1_2[:, 1]

        grad_P2 = torch.autograd.grad(P2.sum(), input_int, create_graph=True)[0]
        P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1]

        grad_P2_1 = torch.autograd.grad(P2_1.sum(), input_int, create_graph=True)[0]
        P2_11, P2_12 = grad_P2_1[:, 0], grad_P2_1[:, 1]
        grad_P2_2 = torch.autograd.grad(P1_2.sum(), input_int, create_graph=True)[0]
        P2_21, P2_22 = grad_P2_2[:, 0], grad_P2_2[:, 1]

        residual_PDE_1 = -0.5841 * varphi_11 - 0.5841 * varphi_22 + P1_1 + P2_2
        residual_PDE_2 = (P1 - P1_previous) / self.delta_t - (
                -2 * 0.148 * P1 - 4 * 0.031 * P1 ** 3 + 2 * 0.63 * P1 * P2 ** 2 + 6 * 0.25 * P1 ** 5 + 0.97 * (
                2 * P1 * P2 ** 4 + 4 * P1 ** 3 * P2 ** 2)) + 0.15 * P1_11 - 0.15 * P2_21 + 0.15 * (
                                 P2_12 + P1_22) - varphi_1
        residual_PDE_3 = (P2 - P2_previous) / self.delta_t - (
                -2 * 0.148 * P2 - 4 * 0.031 * P2 ** 3 + 2 * 0.63 * P2 * P1 ** 2 + 6 * 0.25 * P2 ** 5 + 0.97 * (
                2 * P2 * P1 ** 4 + 4 * P2 ** 3 * P1 ** 2)) + 0.15 * (
                                 P2_11 + P1_21) - 0.15 * P1_12 + 0.15 * P2_22 - varphi_2

        return residual_PDE_1.reshape(-1, ), residual_PDE_2.reshape(-1, ), residual_PDE_3.reshape(-1, )

    def compute_bc_residual(self, input_bc):
        input_bc.requires_grad = True
        u = self.approximate_solution(input_bc)
        varphi = u[:, 2].reshape(-1, 1)
        residual_varphi = varphi

        return residual_varphi.reshape(-1, )

    def compute_bcU_residual(self, input_bc):

        u = self.approximate_solution(input_bc)
        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)

        grad_P1 = torch.autograd.grad(P1.sum(), input_bc, create_graph=True)[0]
        P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1]
        grad_P2 = torch.autograd.grad(P2.sum(), input_bc, create_graph=True)[0]
        P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1]

        residual_U_1 = P1_1 * 0 + P1_2 * 1
        residual_U_2 = P2_1 * 0 + P2_2 * 1

        return residual_U_1.reshape(-1, ), residual_U_2.reshape(-1, )

    def compute_bcD_residual(self, input_bc):

        u = self.approximate_solution(input_bc)
        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)

        grad_P1 = torch.autograd.grad(P1.sum(), input_bc, create_graph=True)[0]
        P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1]
        grad_P2 = torch.autograd.grad(P2.sum(), input_bc, create_graph=True)[0]
        P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1]

        residual_D_1 = P1_1 * 0 + P1_2 * (-1)
        residual_D_2 = P2_1 * 0 + P2_2 * (-1)

        return residual_D_1.reshape(-1, ), residual_D_2.reshape(-1, )

    def compute_bcL_residual(self, input_bc):

        u = self.approximate_solution(input_bc)
        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)

        grad_P1 = torch.autograd.grad(P1.sum(), input_bc, create_graph=True)[0]
        P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1]
        grad_P2 = torch.autograd.grad(P2.sum(), input_bc, create_graph=True)[0]
        P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1]

        residual_L_1 = P1_1 * (-1) + P1_2 * 0
        residual_L_2 = P2_1 * (-1) + P2_2 * 0

        return residual_L_1.reshape(-1, ), residual_L_2.reshape(-1, )

    def compute_bcR_residual(self, input_bc):

        u = self.approximate_solution(input_bc)
        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)

        grad_P1 = torch.autograd.grad(P1.sum(), input_bc, create_graph=True)[0]
        P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1]
        grad_P2 = torch.autograd.grad(P2.sum(), input_bc, create_graph=True)[0]
        P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1]

        residual_R_1 = P1_1 * 1 + P1_2 * 0
        residual_R_2 = P2_1 * 1 + P2_2 * 0

        return residual_R_1.reshape(-1, ), residual_R_2.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, inp_train_int, verbose=True):
        r_int_1, r_int_2, r_int_3 = self.compute_pde_residual(inp_train_int)
        r_sb_varphi = self.compute_bc_residual(inp_train_sb)
        r_sbU_1, r_sbU_2 = self.compute_bcU_residual(inp_train_sb[0:self.n_sb, :])
        r_sbD_1, r_sbD_2 = self.compute_bcD_residual(inp_train_sb[self.n_sb:2 * self.n_sb, :])
        r_sbL_1, r_sbL_2 = self.compute_bcL_residual(inp_train_sb[2 * self.n_sb:3 * self.n_sb, :])
        r_sbR_1, r_sbR_2 = self.compute_bcR_residual(inp_train_sb[3 * self.n_sb:, :])

        loss_sb = torch.mean(abs(r_sb_varphi) ** 2) + torch.mean(
            abs(r_sbU_1) ** 2) + torch.mean(abs(r_sbU_2) ** 2) + torch.mean(
            abs(r_sbD_1) ** 2) + torch.mean(abs(r_sbD_2) ** 2) + torch.mean(
            abs(r_sbL_1) ** 2) + torch.mean(abs(r_sbL_2) ** 2) + torch.mean(
            abs(r_sbR_1) ** 2) + torch.mean(abs(r_sbR_2) ** 2)

        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)

        loss = torch.log10(loss_sb + loss_int)

        if verbose:
            print("Total loss: ", round(loss.item(), 4),
                  "| BC Loss: ", round(torch.log10(loss_sb).item(), 4),
                  "| PDE Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    def calculate_u_end(self, input_int):
        u_end = self.approximate_solution(input_int)
        return u_end

    def fit(self, num_epochs, optimizer, verbose=True):
        history = []
        u_end = None  # 初始化 u_end

        # 记录开始时间
        start_time = time.time()

        for epoch in range(num_epochs):
            if verbose:
                print("################################ ", epoch, " ################################")

            # Iterate through batches of training_set_int and training_set_sb
            for j, (batch_sb, batch_int) in enumerate(zip(self.training_set_sb, self.training_set_int)):
                inp_train_sb = batch_sb[0].to(self.device)
                inp_train_int = batch_int[0].to(self.device)

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, inp_train_int, verbose=verbose)
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

                # 在这里计算 u_end
                u_end = self.calculate_u_end(inp_train_int)

        # 记录结束时间
        end_time = time.time()

        # 计算执行时间
        execution_time = end_time - start_time

        print('Final Loss: ', history[-1])
        print('Execution Time: {:.2f} seconds'.format(execution_time))
        return history, u_end

    def save_checkpoint(self):
        '''save model and optimizer'''
        torch.save({
            'model_state_dict': self.approximate_solution.state_dict()
        }, self.save_dir)

    def load_checkpoint(self):
        '''load model and optimizer'''
        checkpoint = torch.load(self.pre_model_save_path)
        self.approximate_solution.load_state_dict(checkpoint['model_state_dict'])
        print('Pretrained model loaded!')


n_int = 10000
n_sb = 200
delta_t = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
pre_model_save_path = None
save_path = './results/ADAM_test_1.pt'

networks = []  # List to store networks for each time interval
time_intervals = []  # List to store corresponding time intervals
initial_data = pd.read_csv('initial_0.csv').values
u_previous = torch.tensor(initial_data, dtype=torch.float32)

for i in range(10):
    start_time = i * delta_t
    end_time = (i + 1) * delta_t
    time_intervals.append((start_time, end_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create PINN object with previous solution
    Temporal_PINN = Pinns(n_int, n_sb, save_path, pre_model_save_path, device, delta_t, u_previous)
    Temporal_PINN.approximate_solution.to(device)
    n_epochs = 1000
    optimizer_LBFGS = optim.LBFGS(Temporal_PINN.approximate_solution.parameters(),
                                  lr=float(0.5),
                                  max_iter=1000,
                                  max_eval=50000,
                                  history_size=150,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(Temporal_PINN.approximate_solution.parameters(),
                                lr=float(0.001))

    print(f"Training network for time interval [{i * Temporal_PINN.delta_t}, {(i + 1) * Temporal_PINN.delta_t}]")
    _, u_end = Temporal_PINN.fit(num_epochs=n_epochs, optimizer=optimizer_ADAM, verbose=True)  # Fit the PINN

    # Save the current solution for the next time step
    u_previous = u_end
    u_previous = u_previous.to(device)

    #print(u_previous.shape)
    networks.append(Temporal_PINN)

    # Save u_previous to i_output.csv
    output_filename = f"{i+1}_output.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(u_previous.detach().cpu().numpy())

