#torch implementation of lbfgs-bm

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

sys.path.append('../..')

from models.UGD.rprop import Rprop
from models.UGD.cg_optim import cg
from Basis.Basis_State import Mea_basis, State
from evaluation.Fidelity import Fid
from Basis.Loss_Function import MLE_loss, LS_loss, CF_loss
from Basis.Basic_Function import qmt_torch, get_default_device, proj_spectrahedron_torch, qmt_matrix_torch

class lbfgs_nn(nn.Module):

  def __init__(self, na_state, 
                 n_qubits,
                 P_idxs,
                 M):
    super().__init__()
                   
    self.N = n_qubits
    self.P_idxs = P_idxs
    self.M = M
    self.device = M.device 
    self.rank_recon = np.maximum(1, int(2**n_qubits/4))           

    d = 2**n_qubits
    params = torch.randn((2, d, self.rank_recon), requires_grad=True).to(torch.float32)
    self.params = nn.Parameter(params)

  def forward(self):
    self.rho = self.Rho()
    P_out = self.Measure_rho()  # perfect measurement
    return P_out

  def Rho(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    rho = torch.matmul(U, U.T.conj())
    rho = rho / torch.trace(rho)
    return rho

  def Measure_rho(self):
    """Born's Rule"""
    self.rho = self.rho.to(torch.complex64)
    P_all = qmt_torch(self.rho, [self.M] * self.N)

    P_real = P_all[self.P_idxs]
    return P_real

class lbfgs():
  def __init__(self, na_state, generator, P_star, learning_rate=0.01):

    super().__init__
    
    self.generator = generator
    self.P_star = P_star
    
    self.criterion = MLE_loss

    self.optim = optim.LBFGS(self.generator.parameters(), lr=0.1, max_iter=1e3, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)

  def train(self, epochs, fid, result_save):
    """Net training"""
    # self.sche = optim.lr_scheduler.StepLR(self.optim, step_size=1500, gamma=0.2)

    pbar = tqdm(range(epochs), mininterval=0.01)
    epoch = 0
    time_all = 0
    for i in pbar:
        epoch += 1
        time_b = perf_counter()

        self.generator.train()

        def closure():
            self.optim.zero_grad()
            data = self.P_star
            P_out = self.generator()
            loss = self.criterion(P_out, data)
            assert torch.isnan(loss) == 0, print('loss is nan', loss)
            loss.backward()
            return loss

        self.optim.step(closure)
        # self.sche.step()

        time_e = perf_counter()
        time_all += time_e - time_b

        # show and save
        if epoch % 10 == 0 or epoch == 1:
            loss = closure().item()
            self.generator.eval()
            with torch.no_grad():
                rho = self.generator.rho
                rho /= torch.trace(rho)

                Fq = fid.Fidelity(rho)

                result_save['time'].append(time_all)
                result_save['epoch'].append(epoch)
                result_save['Fq'].append(Fq)
                pbar.set_description(
                    "LBFGS_BM loss {:.10f} | Fq {:.8f} | time {:.5f}".format(loss, Fq, time_all))

            if Fq >= 0.9999:
                break

    pbar.close()
