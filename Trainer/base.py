import torch
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from .model import NeuralNet
from .model import MultiVariatePoly  
from tqdm import tqdm
import time
from copy import deepcopy
import os
from .utils import Plot2D
from .utils import StrgridEngine


class Pinns:
    def __init__(self, config, u_previous_):
        self.config = config
        self.pre_model_save_path = config.pre_model_save_path
        self.save_dir = config.save_path

        self.n_int = config.n_int
        self.n_sb = config.n_sb
        self.delta_t = config.delta_t
        self.optimizer_name = config.optimizer

        self.device = config.device
        self.u_previous = u_previous_.to(self.device)
        

        #PDE parameters
        self.L = 100
        self.alpha_1 = -0.148
        self.alpha_11 = -0.031
        self.alpha_12 = 0.63
        self.alpha_111 = 0.25
        self.alpha_112 = 0.97
        self.epsilon_0 = 0.5841
        self.g = 0.15

        self.domain_extrema = torch.tensor([[0, 30],  # x dimension
                                            [0, 20]])  # y dimension

        # Number of space dimensions
        self.space_dimensions = 2

        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=3,
                                              n_hidden_layers=self.config.n_hidden_layers,
                                              neurons=self.config.neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42).to(self.device)
     #   self.writer = SummaryWriter(log_dir="training_log")
     #   self.tags = ["total_loss","pde_loss", "bc_loss","learning_rate"]

        '''self.approximate_solution = MultiVariatePoly(3, 3)'''

        if self.pre_model_save_path:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=2)

        self.strueng = StrgridEngine(dimension=2, grid_size=(30, 20))

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_int = self.assemble_datasets()

        # Optimizer
        self.init_optimizer()

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
        input_int = self.convert(self.strueng.generate_structure_grid())
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
        grad_P2_2 = torch.autograd.grad(P2_2.sum(), input_int, create_graph=True)[0]
        P2_21, P2_22 = grad_P2_2[:, 0], grad_P2_2[:, 1]

        residual_PDE_1 = -self.epsilon_0 * varphi_11 - self.epsilon_0 * varphi_22 + P1_1 + P2_2

        residual_PDE_2 = (1 / self.L) * (P1 - P1_previous) / self.delta_t - (
                + 2 * self.alpha_1 * P1 + 4 * self.alpha_11 * P1 ** 3 + 2 * self.alpha_12 * P1 * P2 ** 2
                + 6 * self.alpha_111 * P1 ** 5 + self.alpha_112 * (
                            2 * P1 * P2 ** 4 + 4 * P1 ** 3 * P2 ** 2)) + self.g * P1_11 + self.g * P1_22 - varphi_1

        residual_PDE_3 = (1 / self.L) * (P2 - P2_previous) / self.delta_t - (
                + 2 * self.alpha_1 * P2 + 4 * self.alpha_11 * P2 ** 3 + 2 * self.alpha_12 * P2 * P1 ** 2
                + 6 * self.alpha_111 * P2 ** 5 + self.alpha_112 * (
                            2 * P2 * P1 ** 4 + 4 * P2 ** 3 * P1 ** 2)) + self.g * P2_11 + self.g * P2_22 - varphi_2


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

        #loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)
        loss_int_1 = torch.mean(abs(r_int_1) ** 2)
        loss_int_2 = torch.mean(abs(r_int_2) ** 2)
        loss_int_3 = torch.mean(abs(r_int_3) ** 2)

        loss = loss_sb + loss_int_1 + loss_int_2 + loss_int_3

        if verbose:
            print("Total loss: ", round(torch.log10(loss).item(), 4),
                  "| BC Loss: ", round(torch.log10(loss_sb).item(), 4),
                  "| PDE_1 Loss: ", round(torch.log10(loss_int_1).item(), 4),
                  "| PDE_2 Loss: ", round(torch.log10(loss_int_2).item(), 4),
                  "| PDE_3 Loss: ", round(torch.log10(loss_int_3).item(), 4))

        return loss, loss_sb, loss_int_1, loss_int_2, loss_int_3

    def init_optimizer(self):
        '''Initialize optimizer'''
        if self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=float(0.5), max_iter=self.config.max_iter,
                                               max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps,
                                               history_size=150, line_search_fn="strong_wolfe")
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=self.config.lr)

        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")
        
        # init scheduler
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.scheduler_step_size, gamma=self.config.scheduler_gamma)
            if self.config.optimizer == "lbfgs":
                raise NotImplementedError(f"Scheduler not implemented for optimizer {self.config.optimizer}")
        else:
            self.scheduler = None
    
    def fit(self, num_epochs, max_iter, lr, verbose=True):
        '''Train process'''

        # Training preparation
        self.approximate_solution.train()
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        losses_sb = []
        losses_int_1 = []
        losses_int_2 = []
        losses_int_3 = []

        epoch = {
            "lbfgs": max_iter,
            "adam": num_epochs
        }[self.optimizer_name]
        #pbar = tqdm(range(epoch), desc = 'Epoch', colour='blue')


        u_end = None  # 初始化 u_end

        def train_batch(batch_sb, batch_int):
            inp_train_sb = batch_sb[0].to(self.device)
            inp_train_int = batch_int[0].to(self.device)

            def closure():
                self.optimizer.zero_grad()
                loss, loss_sb, loss_int_1, loss_int_2, loss_int_3 = self.compute_loss(inp_train_sb, inp_train_int, verbose=verbose)
                # backpropragation
                loss.backward()
                
                # recording
                losses.append(loss.item())
                losses_int_1.append(loss_int_1.item())
                losses_int_2.append(loss_int_2.item())
                losses_int_3.append(loss_int_3.item())
                losses_sb.append(loss_sb.item())
                return loss
            
            return closure

        # training 
        if self.optimizer_name == "lbfgs":
            pbar = tqdm(total=len(self.training_set_sb), desc='Batch', colour='blue') # Progress bar for LBFGS based on batches
            #optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
            
            for j, (batch_sb, batch_int) in enumerate(zip(self.training_set_sb, self.training_set_int)):
                self.optimizer.step(closure=train_batch(batch_sb, batch_int))
            pbar.set_postfix(loss=losses[-1])
            pbar.update(1)
            pbar.close()

        elif self.optimizer_name == "adam":
            pbar = tqdm(total=num_epochs, desc='Epoch', colour='blue') # Progress bar for Adam based on epochs
            #optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=lr)
            
            for ep in range(num_epochs):
                for j, (batch_sb, batch_int) in enumerate(zip(self.training_set_sb, self.training_set_int)):

                    train_batch(batch_sb, batch_int)()
                    self.optimizer.step()
                    if self.config.use_scheduler:
                        self.scheduler.step()

                    #save model
                    if losses[-1] < best_loss:
                        best_epoch = ep
                        best_loss = losses[-1]
                        best_state = deepcopy(self.approximate_solution.state_dict())

                        best_loss = losses[-1]
                pbar.set_postfix(loss=np.mean(losses[-len(self.training_set_sb):]))
                pbar.update(1)
            pbar.close()
            
            self.approximate_solution.load_state_dict(best_state)
            self.save_checkpoint()

        # plot prediction results
        with torch.no_grad():
            u_end = self.approximate_solution(list(self.training_set_int)[0][0].to(self.device))
        
        #Plot2D.Quiver2D(nodecoords=np.array(list(self.training_set_int)[0][0]), sol=u_end.cpu().numpy(), savefig=True, figname='Result')

        # plot losses vs epoch
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.plot(np.arange(len(losses_int_1)), losses_int_1, label="loss_int_1")
        ax.plot(np.arange(len(losses_int_2)), losses_int_2, label="loss_int_2")
        ax.plot(np.arange(len(losses_int_3)), losses_int_3, label="loss_int_3")
        ax.plot(np.arange(len(losses_sb)), losses_sb, label="loss_sb")
        if best_epoch != -1:
            ax.scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_yscale('log')
        plt.savefig(f'loss.png')
        

        return u_end

    def testing():
        pass

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

class Pinns2:
    def __init__(self, config):
        self.config = config
        self.pre_model_save_path = config.pre_model_save_path
        self.save_dir = config.save_path

        self.n_int = config.n_int
        self.n_sb = config.n_sb
        self.n_tb = config.n_tb
        #self.delta_t = config.delta_t
        self.optimizer_name = config.optimizer

        self.device = config.device

        # PDE parameters
        self.L = 100
        self.alpha_1 = -0.148
        self.alpha_11 = -0.031
        self.alpha_12 = 0.63
        self.alpha_111 = 0.25
        self.alpha_112 = 0.97
        self.epsilon_0 = 0.5841
        self.g = 0.15

        self.domain_extrema = torch.tensor([[0, 30],  # x dimension
                                            [0, 20],  # y dimension
                                            [0, 5]])   # t dimension

        # Number of space dimensions
        self.space_dimensions = 2

        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=3,
                                              n_hidden_layers=self.config.n_hidden_layers,
                                              neurons=self.config.neurons,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42).to(self.device)
        #   self.writer = SummaryWriter(log_dir="training_log")
        #   self.tags = ["total_loss","pde_loss", "bc_loss","learning_rate"]

        '''self.approximate_solution = MultiVariatePoly(3, 3)'''

        if self.pre_model_save_path:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=3)

        #self.strueng = StrgridEngine(dimension=2, grid_size=(30, 20))

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

        # Optimizer
        self.init_optimizer()

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
        print(input_sb.shape)
        print(input_sb)
        return input_sb

    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[2, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 2] = torch.full(input_tb[:, 2].shape, t0)
        print(input_tb.shape)
        print(input_tb)
        return input_tb

    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        print(input_int.shape)
        print(input_int)
        return input_int

    def assemble_datasets(self):
        input_sb = self.add_spatial_boundary_points()   # S_sb
        input_tb = self.add_temporal_boundary_points()  # S_tb
        input_int = self.add_interior_points()          # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb),
                                     batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb), batch_size=self.n_tb,
                                     shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int), batch_size=self.n_int,
                                      shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def compute_ic_residual(self, input_tb):
        input_tb.requires_grad = True
        u = self.approximate_solution(input_tb)
        P1 = u[:, 0].reshape(-1, 1)
        P2 = u[:, 1].reshape(-1, 1)
        varphi = u[:, 2].reshape(-1, 1)

        residual_P = P1 ** 2 + P2 ** 2 - 0.7 ** 2
        residual_varphi = varphi
        return residual_varphi.reshape(-1, ), residual_P.reshape(-1, )

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
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
        P1_1, P1_2, P1_t = grad_P1[:, 0], grad_P1[:, 1], grad_P1[:, 2]

        grad_P1_1 = torch.autograd.grad(P1_1.sum(), input_int, create_graph=True)[0]
        P1_11, P1_12 = grad_P1_1[:, 0], grad_P1_1[:, 1]
        grad_P1_2 = torch.autograd.grad(P1_2.sum(), input_int, create_graph=True)[0]
        P1_21, P1_22 = grad_P1_2[:, 0], grad_P1_2[:, 1]

        grad_P2 = torch.autograd.grad(P2.sum(), input_int, create_graph=True)[0]
        P2_1, P2_2, P2_t = grad_P2[:, 0], grad_P2[:, 1], grad_P2[:, 2]

        grad_P2_1 = torch.autograd.grad(P2_1.sum(), input_int, create_graph=True)[0]
        P2_11, P2_12 = grad_P2_1[:, 0], grad_P2_1[:, 1]
        grad_P2_2 = torch.autograd.grad(P2_2.sum(), input_int, create_graph=True)[0]
        P2_21, P2_22 = grad_P2_2[:, 0], grad_P2_2[:, 1]

        residual_PDE_1 = -self.epsilon_0 * varphi_11 - self.epsilon_0 * varphi_22 + P1_1 + P2_2

        residual_PDE_2 = (1 / self.L) * P1_t - (
                + 2 * self.alpha_1 * P1 + 4 * self.alpha_11 * P1 ** 3 + 2 * self.alpha_12 * P1 * P2 ** 2
                + 6 * self.alpha_111 * P1 ** 5 + self.alpha_112 * (
                        2 * P1 * P2 ** 4 + 4 * P1 ** 3 * P2 ** 2)) + self.g * P1_11 + self.g * P1_22 - varphi_1

        residual_PDE_3 = (1 / self.L) * P2_t - (
                + 2 * self.alpha_1 * P2 + 4 * self.alpha_11 * P2 ** 3 + 2 * self.alpha_12 * P2 * P1 ** 2
                + 6 * self.alpha_111 * P2 ** 5 + self.alpha_112 * (
                        2 * P2 * P1 ** 4 + 4 * P2 ** 3 * P1 ** 2)) + self.g * P2_11 + self.g * P2_22 - varphi_2

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
    def compute_loss(self, inp_train_sb, inp_train_tb, inp_train_int, verbose=True):
        r_int_1, r_int_2, r_int_3 = self.compute_pde_residual(inp_train_int)
        r_sb_varphi = self.compute_bc_residual(inp_train_sb)
        r_sbU_1, r_sbU_2 = self.compute_bcU_residual(inp_train_sb[0:self.n_sb, :])
        r_sbD_1, r_sbD_2 = self.compute_bcD_residual(inp_train_sb[self.n_sb:2 * self.n_sb, :])
        r_sbL_1, r_sbL_2 = self.compute_bcL_residual(inp_train_sb[2 * self.n_sb:3 * self.n_sb, :])
        r_sbR_1, r_sbR_2 = self.compute_bcR_residual(inp_train_sb[3 * self.n_sb:, :])
        r_tb_varphi, r_tb_P = self.compute_ic_residual(inp_train_tb)

        loss_sb = torch.mean(abs(r_sb_varphi) ** 2) + torch.mean(
            abs(r_sbU_1) ** 2) + torch.mean(abs(r_sbU_2) ** 2) + torch.mean(
            abs(r_sbD_1) ** 2) + torch.mean(abs(r_sbD_2) ** 2) + torch.mean(
            abs(r_sbL_1) ** 2) + torch.mean(abs(r_sbL_2) ** 2) + torch.mean(
            abs(r_sbR_1) ** 2) + torch.mean(abs(r_sbR_2) ** 2)

        loss_tb = torch.mean(abs(r_tb_varphi) ** 2) + torch.mean(abs(r_tb_P) ** 2)
        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2)
        #loss_int_1 = torch.mean(abs(r_int_1) ** 2)
        #loss_int_2 = torch.mean(abs(r_int_2) ** 2)
        #loss_int_3 = torch.mean(abs(r_int_3) ** 2)

        loss = loss_sb + loss_int + loss_tb

        if verbose:
            print("Total loss: ", round(torch.log10(loss).item(), 4),
                  "| BC Loss: ", round(torch.log10(loss_sb).item(), 4),
                  "| IC Loss: ", round(torch.log10(loss_tb).item(), 4),
                  "| PDE Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss, loss_sb, loss_tb, loss_int

    def init_optimizer(self):
        '''Initialize optimizer'''
        if self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=float(0.5),
                                               max_iter=self.config.max_iter,
                                               max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps,
                                               history_size=150, line_search_fn="strong_wolfe")
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=self.config.lr)

        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        # init scheduler
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.scheduler_step_size,
                                                             gamma=self.config.scheduler_gamma)
            if self.config.optimizer == "lbfgs":
                raise NotImplementedError(f"Scheduler not implemented for optimizer {self.config.optimizer}")
        else:
            self.scheduler = None

    def fit(self, num_epochs, max_iter, lr, verbose=True):
        '''Train process'''

        # Training preparation
        self.approximate_solution.train()
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        losses_sb = []
        losses_tb = []
        losses_int = []

        epoch = {
            "lbfgs": max_iter,
            "adam": num_epochs
        }[self.optimizer_name]
        # pbar = tqdm(range(epoch), desc = 'Epoch', colour='blue')

        def train_batch(batch_sb, batch_tb, batch_int):
            inp_train_sb = batch_sb[0].to(self.device)
            inp_train_tb = batch_tb[0].to(self.device)
            inp_train_int = batch_int[0].to(self.device)

            def closure():
                self.optimizer.zero_grad()
                loss, loss_sb, loss_tb, loss_int = self.compute_loss(
                    inp_train_sb, inp_train_tb, inp_train_int, verbose=verbose)
                # backpropragation
                loss.backward()
                # recording
                losses.append(loss.item())
                losses_sb.append(loss_sb.item())
                losses_tb.append(loss_tb.item())
                losses_int.append(loss_int.item())
                if self.config.optimizer == "lbfgs":
                    pbar.set_postfix(loss=losses[-1])
                    pbar.update(1)

                return loss

            return closure

        # training
        if self.optimizer_name == "lbfgs":
            pbar = tqdm(total=self.config.max_iter, desc='Batch',
                        colour='blue')  # Progress bar for LBFGS based on batches
            # optimizer = torch.optim.LBFGS(self.approximate_solution.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

            for j, (batch_sb, batch_tb, batch_int) in enumerate(zip(
                    self.training_set_sb, self.training_set_tb, self.training_set_int)):
                self.optimizer.step(closure=train_batch(batch_sb, batch_tb, batch_int))
            pbar.close()
            self.save_checkpoint()

        elif self.optimizer_name == "adam":
            pbar = tqdm(total=num_epochs, desc='Epoch', colour='blue')  # Progress bar for Adam based on epochs
            # optimizer = torch.optim.Adam(self.approximate_solution.parameters(), lr=lr)

            for ep in range(num_epochs):
                for j, (batch_sb, batch_tb, batch_int) in enumerate(zip(
                        self.training_set_sb, self.training_set_tb, self.training_set_int)):

                    train_batch(batch_sb, batch_tb, batch_int)()
                    self.optimizer.step()
                    if self.config.use_scheduler:
                        self.scheduler.step()

                    # save model
                    if losses[-1] < best_loss:
                        best_epoch = ep
                        best_loss = losses[-1]
                        best_state = deepcopy(self.approximate_solution.state_dict())

                        best_loss = losses[-1]
                pbar.set_postfix(loss=np.mean(losses[-len(self.training_set_sb):]))
                pbar.update(1)
            pbar.close()

            self.approximate_solution.load_state_dict(best_state)
            self.save_checkpoint()

        # Plot2D.Quiver2D(nodecoords=np.array(list(self.training_set_int)[0][0]), sol=u_end.cpu().numpy(), savefig=True, figname='Result')

        # plot losses vs epoch
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.plot(np.arange(len(losses_sb)), losses_sb, label="loss_sb")
        ax.plot(np.arange(len(losses_tb)), losses_tb, label="loss_tb")
        ax.plot(np.arange(len(losses_int)), losses_int, label="loss_int")

        if best_epoch != -1:
            ax.scatter([best_epoch], [best_loss], c='r', marker='o', label="best loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_yscale('log')
        plt.savefig(f'loss.png')

        return losses

    def testing(self):
        pass

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

    def plotting(self):
        t = torch.linspace(0, 5, 50)
        x = torch.linspace(0, 30, 60)
        y = torch.linspace(0, 20, 40)
        grid = torch.meshgrid(t, x, y, indexing='ij')
        inputs = torch.stack(grid, dim=-1).reshape(-1, 3)
        outputs = self.approximate_solution(inputs).detach().numpy()
        P1 = outputs[:, 0]
        P2 = outputs[:, 1]

        magnitude = np.sqrt(P1 ** 2 + P2 ** 2)
        P1_normalized = P1 / magnitude
        P2_normalized = P2 / magnitude

        for i, t_val in enumerate(t):
            if i % 5 == 0:
                xx, yy = np.meshgrid(x, y, indexing='ij')
                P1_slice = P1_normalized[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                P2_slice = P2_normalized[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                magnitude_slice = magnitude[i * xx.size:(i + 1) * xx.size].reshape(xx.shape)
                plt.figure(figsize=(8, 5), dpi=150)
                plt.quiver(xx, yy, P1_slice, P2_slice, magnitude_slice, angles='xy', scale_units='xy', scale=1,
                           cmap='jet')
                plt.colorbar(label='Magnitude')
                plt.title(f"Velocity Vector Plot at t = {t_val}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig('./result_plot/P_vector_test_' + str(i))