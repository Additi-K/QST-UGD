# file to run low memory version of QST for large qubit systems

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

from Basis.Basis_State import Mea_basis, State
from Basis.Basic_Function import get_default_device
from evaluation.Fidelity import Fid
from datasets.dataset import Dataset_P, Dataset_sample, Dataset_sample_P
from models.others.LRE import LRE
from models.others.qse_apg import qse_apg
from models.others.iMLE import iMLE
from models.UGD.ugd import UGD_nn, UGD
from models.others.lbfgs_bm import lbfgs_nn, lbfgs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def optimized_order(n, i):
    base_pattern = np.concatenate([
        np.full(2**i, 2**i),
        np.full(2**i, -2**i)
    ])  # Create base pattern

    order = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
    return order  # No need for extra multiplications

def optimized_scaleY(n, i):
    base_pattern = np.concatenate([
        np.full(2**i, -1j),
        np.full(2**i, 1j)
    ])  # Create base pattern

    scale = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
    return scale  # No need for extra multiplications

def optimized_scaleZ(n, i):
    base_pattern = np.concatenate([
        np.full(2**i, 1),
        np.full(2**i, -1)
    ])  # Create base pattern

    scale = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
    return scale  # No need for extra multiplications

def testX(u, N, i):
  n = int(np.log2(N))

  order = optimized_order(n, i)

  idx = np.arange(N)

  return u[idx+order]

def testZ(u, N, i):
  n = int(np.log2(N))

  scale = optimized_scaleZ(n, i)
  u *= scale[:, np.newaxis]

  return u


def testY(u, N, i):
  n = int(np.log2(N))

  scale = optimized_scaleY(n, i)
  order = optimized_order(n, i)

  idx = np.arange(N)

  return scale[:, np.newaxis]* u[idx+order]

def testI(u, N, i):
  return u

PauliFcn_map = {0:testI, 1:testZ, 2:testX, 3:testY}

def cal_P_lowmem(x, lst):
  if x.ndim == 1: x = x.reshape( (-1,1) ) # make sure it's a column vector
    r = x.shape[1]

    n_qubits = int( np.log2( x.shape[0] ) )

    # Au = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.complex64)
    m = ((lst.shape[0]
    P = np.zeros((m,))

    for i in range(m):
      u = x.copy()  
      for ni, p in enumerate( reversed(int2lst(lst[i], n_qubits))):
          u = PauliFcn_map[p]( u, 2**n_qubits, ni)
      P[i] = np.real_if_close(np.vdot(x, u))

    return np.concatenate( [0.5*(P[0]+[1:]), 0.5*(P[0]-[1:])] )
  

class lbfgs_nn_lowmem(nn.Module):
  
  def __init__(self, n_qubits, rank, POVM_set):
    super().__init__()

    self.N = n_qubits
    self.POVM_set = POVM_set
    self.rank = rank

    d = 2**n_qubits
    params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
    self.params = nn.Parameter(params)

  def forward(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    U = U.cpu().numpy()
    P_out = cal_P_lowmem(U, self.POVM_set)
    
    return P_out


class lbfgs():
  def __init__(self, generator, P_star, learning_rate=0.01):
      self.generator = generator
      self.P_star = P_star
      self.criterion = MLE_loss

      self.optim = optim.LBFGS(self.generator.parameters(), lr=0.1, max_iter=1000, 
                               tolerance_grad=1e-07, tolerance_change=1e-09, 
                               history_size=10, line_search_fn=None)
      
      self.overhead_t = 0
      self.epoch = 0
      self.time_all = 0 

  def track_parameters(self, loss, fid, result_save):
      """Callback to store parameter updates (excluding computation time)."""

      start_overhead = perf_counter()  # Start timing overhead
      self.generator.eval()

      with torch.no_grad():
          penalty = 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2

          result_save['epoch'].append(self.epoch)
          # result_save['Fq'].append(Fq)
          result_save['loss'].append(loss.item() - penalty)
          self.epoch += 1

      self.overhead_t = perf_counter() - start_overhead  # ✅ Correct overhead timing

  def train(self, epochs, fid, result_save):
      """Net training."""
      pbar = tqdm(range(1), mininterval=0.01)
      epoch = 0

      for _ in pbar:
          epoch += 1
          

          self.generator.train()

          def closure():
              self.generator.train()
              time_b = perf_counter()
              self.optim.zero_grad()
              P_out = self.generator()
              loss = self.criterion(P_out, self.P_star)
              loss += 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2
              
              assert not torch.isnan(loss), "Loss is NaN" 
              loss.backward()
              self.track_parameters(loss, fid, result_save)
              # Update tracking (exclude overhead from time_all)
              raw_t = perf_counter()
              self.time_all += raw_t - time_b - self.overhead_t
              result_save['time'].append(self.time_all)

              return loss

          self.optim.step(closure)

      # Print tracked updates
      for i, (l, t) in enumerate(zip(result_save['loss'], result_save['time'])):
          print("LBFGS_BM loss {:.10f} | time {:.5f}".format(l, t))

      pbar.close()

    
    
  

