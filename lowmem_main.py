import torch
import numpy as np
from scipy import sparse
import scipy.linalg as sla
from scipy.linalg import norm
trace = lambda rho : np.real_if_close(np.trace(rho))
from time import time
import matplotlib.pyplot as plt
import pickle

from models.others.lbfgs_bm import lbfgs_nn

#######################################################
##low memory implementation for probability calculation
#######################################################


def int2lst(j, nQubits ):
    """ input: 0 <= j < 4^nQubits,
    where j represents one of the measurements.
    We'll take the 4-nary representation of j
    and return it as a list of length nQubits
    Ex: j = 19 and nQubits = 4 returns [0, 0, 1, 0, 3]
    """
    pad = nQubits-np.floor(np.log(np.maximum(j,.3))/np.log(4))-1
    lst = np.base_repr( j, base=4, padding=int(pad) )

    return [int(i) for i in lst ]
    # return list(map(int, lst))

def optimized_orderX(n, i):
    base_pattern = np.concatenate([
        np.full(2**i, 2**i),
        np.full(2**i, -2**i)
    ])  # Create base pattern

    order = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
    return order  # No need for extra multiplications


# def optimized_scaleY(n, i):
#     base_pattern = np.concatenate([
#         np.full(2**i, -1j),
#         np.full(2**i, 1j)
#     ])  # Create base pattern

#     scale = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
#     return scale  # No need for extra multiplications

def optimized_scaleZ(n, i):
    base_pattern = np.concatenate([
        np.full(2**i, 1),
        np.full(2**i, -1)
    ])  # Create base pattern

    scale = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
    return scale  # No need for extra multiplications



def testX(u, N, i):
  n = int(np.log2(N))

  order = optimized_orderX(n, i)

  idx = np.arange(N)

  return u[idx+order]

def testZ(u, N, i):
  n = int(np.log2(N))

  scale = optimized_scaleZ(n, i)
  u *= scale[:, np.newaxis]

  return u


def testY(u, N, i):
  n = int(np.log2(N))

  # scale = [-1j]*2**i + [1j]*2**i
  # scale = np.array(scale*(2**(n-(i+1))))
  # order = [2**i]*2**i + [-2**i]*2**i
  # order = np.array(order*(2**(n-(i+1))))

  scale = -1j*optimized_scaleZ(n, i)
  order = optimized_orderX(n, i)

  idx = np.arange(N)

  return scale[:, np.newaxis]* u[idx+order]

def testI(u, N, i):
  return u

PauliFcn_map = {0:testI, 1:testZ, 2:testX, 3:testY}

def lowmemAu(u, meas):

    """ March 2025, write code without using tensor reshapes...
    This is the building block for other codes, e.g., ones that do tr( A@u@u.T ) = vdot(u,A@u)
        or ones that build this up for a gradient

    To extend this, will need to:
        - allow rank r>1  (we may want to transpose u so that it's r x n not n x r,
                           because we'd want the new code to work on the "fast" dimension)
        - loop over multiple measurements
        - accumulate into a gradient (e.g., we need the frequencies, and the "flag" of +1 or -1 to convert from Pauli to POVM)

    We might also need an adjoint operator, unless we do backpropagation...

    """
    m = len(meas)
    u = u.cpu().numpy()
    y = torch.zeros((m, 1))
    nQubits = int( np.log2( u.shape[0] ) )
    for M in meas:
        v = u.copy()
        for ni,p in enumerate( reversed(int2lst(M, nQubits )) ):
            v = PauliFcn_map[p]( v, 2**nQubits, ni)
        
        y[i] = np.vdot(u.copy(), v)    

    return y


#######################################################
## build a NN for lbfgs for low memory
#######################################################


class lowmem_lbfgs_nn(nn.Module):

  def __init__(self, na_state, 
                 n_qubits,
                 P_idxs,
                 ):
    super().__init__()
                   
    self.N = n_qubits
    self.P_idxs = P_idxs
    self.device = 'cuda' 
    self.rank = np.maximum(1, int(2**n_qubits/4))           

    d = 2**n_qubits
    params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
    self.params = nn.Parameter(params)

  def forward(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    P_out = lowmemAu(U, self.P_idxs)
      
    return P_out

class lowmem_lbfgs():
  
    def __init__(self, na_state, generator, P_star, learning_rate=0.01):
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
          rho = self.generator.rho
          rho /= torch.trace(rho)
          penalty = 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2

          Fq = fid.Fidelity(rho)

          result_save['epoch'].append(self.epoch)
          result_save['Fq'].append(Fq)
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
      for i, (f, l, t) in enumerate(zip(result_save['Fq'], result_save['loss'], result_save['time'])):
          print("LBFGS_BM loss {:.10f} | Fq {:.8f} | time {:.5f}".format(l, f, t))

      pbar.close()






