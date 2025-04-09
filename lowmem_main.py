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

def lowmemAu(meas, u):

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
    if u.ndim == 1: u = u.reshape( (-1,1) ) # make sure it's a column vector
    r = u.shape[1]

    nQubits = int( np.log2( u.shape[0] ) )
    d = nQubits # alias

    Au = torch.zeros((u.shape[0], u.shape[1]), dtype=torch.complex64)
    # if r > 1:
    #     raise NotImplementedError()

    # v = u.copy()

    for ni,p in enumerate( reversed(int2lst(meas, nQubits )) ):

        u = PauliFcn_map[p]( u, 2**nQubits, ni)

    return u


#######################################################
## build a NN for lbfgs for low memory
#######################################################


class lowmem_lbfgs_nn(lbgfs_nn):

  def __init__(self, na_state, 
                 n_qubits,
                 P_idxs,
                 ):
    super().__init__()
                   
    self.N = n_qubits
    self.P_idxs = P_idxs
    self.device = M.device 
    self.rank = np.maximum(1, int(2**n_qubits/4))           

    d = 2**n_qubits
    params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
    self.params = nn.Parameter(params)

  def forward(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    P_out = lowmem_Au(U, self.P_idxs)
    return P_out


