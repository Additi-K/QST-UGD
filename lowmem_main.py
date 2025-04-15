import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from time import time
from tqdm import tqdm
import numpy as np
from scipy import sparse
import scipy.linalg as sla
from scipy.linalg import norm
trace = lambda rho : np.real_if_close(np.trace(rho))


from Basis.Basic_Function import get_default_device
from Basis.Basis_State import Mea_basis, State
from Basis.Loss_Function import MLE_loss
# from models.others.lbfgs_bm import lowmem_lbfgs_nn, lowmem_lbfgs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#######################################################
##low memory implementation for probability calculation
#######################################################

#######################################################
## numpy implementation
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
  if torch.is_tensor(u):
      u_np = u.detach().cpu().numpy()
  
  y = np.zeros((2*m, ))
  
  nQubits = int( np.log2( u_np.shape[0] ) )

  srt = time()

  tr =  np.real_if_close( np.vdot(u_np.copy().T, u_np.copy()) )
  # print('lowmemeAu:', tr)
  
  for i in range(meas.shape[0]):
    v = u_np.copy()
    for ni,p in enumerate( reversed(int2lst(meas[i], nQubits )) ):
        v = PauliFcn_map[p]( v, 2**nQubits, ni)
      
    y[i] = np.real_if_close( np.vdot(u_np.copy().T, v) )  
  # print('time:', time()-srt)
  temp = y[0:m]
  y[0:m] = 0.5*(tr+temp)
  y[m:] = 0.5*(tr-temp)
  
  # return torch.from_numpy(y).to(u.device)
  return y
  

def lowmemAu_all(u, meas):

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
  if torch.is_tensor(u):
      u_np = u.detach().cpu().numpy()
  
  y = np.zeros((m, 1))
  
  nQubits = int( np.log2( u_np.shape[0] ) )

  srt = time()
  for i in range(meas.shape[0]):
      v = u_np.copy()
      for ni,p in enumerate( reversed(int2lst(meas[i], nQubits )) ):
          v = PauliFcn_map[p]( v, 2**nQubits, ni)
      
      y[i] = np.real_if_close( np.vdot(u_np.copy().T, v) )  
  
  print('time:', time()-srt)

  return y  


class TimeExceededException(Exception):
    """Custom exception to stop optimization when time limit is exceeded."""
  pass

class TimingCallback:
  def __init__(self, fun, fidelity, reshape, start_time, timeLimit):
      self.fidelity = fidelity
      self.fun = fun
      self.timeLimit = timeLimit
      self.result_save = {'time_all': [],
                'epoch': [],
                'Fq': [], 
              'loss': []}


  def __call__(self, intermediate_result):
      t1 = time()
      x = np.copy(intermediate_result.x)
      x = self.reshape(x)
    
      # update stats
      self.result_save['epoch'].append(self.cnt)
      self.cnt += 1
    
      fval = self.fun(x)
      self.result_save['loss'].append(fval)
    
      fid = self.fidelity(x)
      self.result_save['Fq'].append(fid)
    
      t = t1 - self.start_time
      self.result_save['time_all'].append(t) 
      
      if self.cnt%10 == 0:
        print("LBFGS_BM loss {:.10f} | Fq {:.8f} | time {:.5f}".format(fval, fid, t))
        # Check if total elapsed time exceeds the limit
        if self.result_save['time_all'][-1] > self.timeLimit:
          print(f"Time limit exceeded: {self.result_save['time_all'][-1]:.2f}s > {self.timeLimit:.2f}s")
          raise TimeExceededException("Time limit exceeded, stopping optimization")



class LBFGS_numpy():
  def __init__(self,  ):
    self.rho_true = param['rho_true']
    self.fun = param['fun']
    self.gradf = param['gradf']
    self.f = param['data'] 
    self.n_qubits = param['n_qubits'] 
    self.rank = param['rank']
    self.povm = meas
    self.timeLimit = 3600
    self.lmbda = 2*np.sum(self.f)
    self.start_time = time()
    self.callback = TimingCallback(self.forward, self.fidelity, self.reshape, self.start_time(), self.timeLimit)
     
    

    # Initialize u
    dim = 2*self.rank*(2)**self.n_qubits
    self.u0 = np.random.randn((dim))
    self.u0 = self.u0/ np.linalg.norm(self.u0)
  
  def forward(self, u):
    # return function value 
    p_out = lowmemAu(u, self.povm)
    
    return -np.dot(self.f, np.log(p_out)) + 0.5* self.lmbda*np.linalg.norm(u)**2


  def gradient(self, u):
    m = self.povm.shape[0]
    grad = np.zeros(u.shape)
    tr =  np.real_if_close( np.vdot(u, u) )
  
    for i in range(m):
      v = u.copy()
      for ni,p in enumerate( reversed(int2lst(povm[i], self.n_qubits )) ):
        v = PauliFcn_map[p]( v, 2**self.n_qubits, ni)
      temp = np.vdot(u, v)  
      grad +=  1.0*(self.f[i]*(u + v)/(tr + temp) + self.f[i+m]*(u - v)/(tr - temp))

    grad *= -2
    grad+= self.lmbda*u

    return grad

  def optimize(self):
    """Run L-BFGS-B to minimize the objective starting from x0."""
    try:
      result = minimize(
          fun=self.forward,
          x0=self.u0,
          jac=self.gradient,             # use analytic gradient
          method='L-BFGS-B',
          callback=self.callback,
          options={
            'disp': False,
            'ftol': 1e-9,
            'gtol': 1e-9,
            'iprint': -1,
            'maxiter': 100,
            'maxfun': 1000,
            'maxcor': 10
              }
      )

    except TimeExceededException as e:
        return self.callback.result_save

    return self.callback.result_save

    
  def reshape(self, u):
    idx = int(u.shape[0]/2)
    reshaped_u = u[0:idx].reshape((2**self.n_qubits, self.rank), order = 'C') + 1j*u[idx:].reshape((2**self.n_qubits, self.rank), order='C')
    return reshaped_u

  def fidelity(self, u):
    return np.linalg.norm(np.vdot(self.true_rho, u))**2            
    
    


#######################################################
## torch version - build a NN for lbfgs for low memory
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
    self.rank = 1     

    d = 2**self.N
    params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
    # Compute L2 norm along the last dimension (rank)
    norm = params.norm(p=2)

    # Normalize params to unit norm
    params_unit = params / norm
    self.params = nn.Parameter(params_unit)
  
  def forward(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    P_out = lowmemAu(U, self.P_idxs)
      
    return P_out

def mle_loss(out, target):
  assert out.shape == target.shape, "Shape mismatch between output and target"
  loss = -target.dot(torch.log(out))
  return loss

def cal_grad(u, x, meas):
  
  m = len(meas)
  if torch.is_tensor(u):
      u_np = u.detach().cpu().numpy()
      x_np = u.detach().cpu().numpy()
  
  y = np.zeros(u_np.shape)
  
  nQubits = int( np.log2( u_np.shape[0] ) )

  tr =  np.real_if_close( np.vdot(u_np.copy().T, u_np.copy()) )
  
  for i in range(meas.shape[0]):
    v = u_np.copy()
    for ni,p in enumerate( reversed(int2lst(meas[i], nQubits )) ):
        v = PauliFcn_map[p]( v, 2**nQubits, ni)
    temp = np.vdot(u_np.copy().T, v)  
    y +=  1.0*(x_np[i]*(u_np + v)/(tr + temp) + x_np[i+m]*(u_np - v)/(tr - temp))

  y *= -2
  y+= 2*np.sum(x_np)*u_np
  
  return torch.from_numpy(y).to(u.device)

class lowmem_lbfgs():
  def __init__(self, na_state, generator, P_star, learning_rate=0.01):
    self.generator = generator
    self.P_star = P_star
    self.criterion = mle_loss

    self.optim = optim.LBFGS(self.generator.parameters(), lr=1e-4, max_iter=100, 
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
      rho = torch.complex(self.generator.params[0,:,:], self.generator.params[1,:,:])
      # print('track:', torch.norm(self.generator.params, p=2))
      penalty = 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2

      Fq = fid.fidelity(rho)

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
    

      # self.generator.train()

      def closure():
        self.generator.train()
        time_b = perf_counter()
        self.optim.zero_grad()
        P_out = self.generator()
        loss = self.criterion(P_out, self.P_star) + 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2
        
        assert not torch.isnan(loss), "Loss is NaN" 
        loss.backward()
        print('u estimated:', torch.complex(self.generator.params[0,:,:], self.generator.params[1,:,:]))
        
        # for name, param in self.generator.named_parameters():
        #   if param.grad is not None:
        #     print(f"Gradient of {name}: {torch.norm(param.grad)}")
        #   else:
        #     print(f"{name} has no gradient.")
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
    print('u estimated:', torch.complex(self.generator.params[0,:,:], self.generator.params[1,:,:]))
    pbar.close()

class Fidelity():

  def __init__(self, rho_star):
    
    self.rho_star = rho_star

  def fidelity(self, sigma):

    return torch.norm(torch.vdot(self.rho_star.view(-1), sigma.view(-1)))**2

###################################################
## setup and run
###################################################
def Dataset_sample_lowmem(na_state, n_qubits, n_samples, P_state,ty_state, rho_star, read_data, P_povm, seed_povm):
    
  all = np.arange(0, 4**n_qubits) 
  pmf = lowmemAu_all(rho_star, all)
  pmf = pmf**2/2**n_qubits

  # number of povms to take
  epsilon = 0.03
  delta = 0.10                                                  
  l = min(int(np.ceil(np.log(1 / delta) / (epsilon ** 2))), len(pmf) - 1)
  meas = np.argsort(pmf[1:], axis=0)[-l:]
  meas = np.sort(meas)

  pmf = 0.5*(pmf[0] + pmf[meas+1]).flatten()

  out1 = np.random.binomial( 1000, np.real_if_close(pmf))
  out2 = 1000 - out1
  out1 = out1/1000
  out2 = out2/1000
  
  out1 = torch.from_numpy(out1).to('cuda')
  out2 = torch.from_numpy(out2).to('cuda')
  # print(out1, out2)
  return (meas+1).flatten() , torch.cat((out1, out2), dim=0)                     

    
    
def Net_train(opt, device, r_path, rho_star=None):
  """
  *******Main Execution Function*******
  """
  torch.cuda.empty_cache()
  print('\nparameter:', opt)

  # ----------file----------
  if os.path.isdir(r_path):
      print('result dir exists, is: ' + r_path)
  else:
      os.makedirs(r_path)
      print('result dir not exists, has been created, is: ' + r_path)

  # ----------rho_star and M----------
  print('\n'+'-'*20+'rho'+'-'*20)
  if rho_star is None:
      state_star, rho_star = State().Get_state_rho(
          opt.na_state, opt.n_qubits, opt.P_state, opt.rank)
  # print(rho_star.shape, state_star.shape)
  if opt.ty_state == 'pure':  # pure state
      rho_star = state_star
  # print('norm of u:', np.linalg.norm(rho_star))
  rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)
  print('true u:', rho_star)
  # ----------data----------
  print('\n'+'-'*20+'data'+'-'*20)
  print('read original data')
  
  print('----read sample data')
  P_idxs, data =  Dataset_sample_lowmem(opt.na_state, opt.n_qubits,
                                                    opt.n_samples, opt.P_state,
                                                    opt.ty_state, rho_star, opt.read_data,
                                                    opt.P_povm, opt.seed_povm)

  in_size = len(data)
  print('data shape:', in_size)
  fid = Fidelity(rho_star)
  

  # ----------------------------------------------QST algorithms----------------------------------------------------
  result_saves = {}

  # ---1: LBFGS with BM low-memory---
  print('\n'+'-'*20+'lbfgs_bm'+'-'*20)
  gen_net = lowmem_lbfgs_nn(opt.na_state, opt.n_qubits, P_idxs).to(torch.float32).to(device)

  net = lowmem_lbfgs(opt.na_state, gen_net, data, opt.lr)
  result_save = {'parser': opt,
                  'time': [],
                  'epoch': [],
                  'Fq': [], 
                'loss': []}
  net.train(opt.n_epochs, fid, result_save)
  result_saves['lbfgs'] = result_save

  return result_saves


if __name__ == '__main__':
  """
  *******Main Function*******
  Given QST perform parameters.
  """
  # ----------device----------
  print('-'*20+'init'+'-'*20)
  device = get_default_device()
  print('device:', device)

  # ----------parameters----------
  print('-'*20+'set parser'+'-'*20)
  parser = argparse.ArgumentParser()
  parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
  parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

  parser.add_argument("--n_qubits", type=int, default=6, help="number of qubits")
  parser.add_argument("--na_state", type=str, default="W_P", help="name of state in library")
  parser.add_argument("--P_state", type=float, default=1.0, help="P of mixed state")
  parser.add_argument("--rank", type=float, default=2**5, help="rank of mixed state")
  parser.add_argument("--ty_state", type=str, default="pure", help="type of state (pure, mixed)")

  parser.add_argument("--noise", type=str, default="noise", help="have or have not sample noise (noise, no_noise)")
  parser.add_argument("--n_samples", type=int, default=int(1e10), help="number of samples")
  parser.add_argument("--P_povm", type=float, default=1, help="possbility of sampling POVM operators")
  parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
  parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

  parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
  parser.add_argument("--lr", type=float, default=0.1, help="optim: learning rate")

  parser.add_argument("--map_method", type=str, default="fac_h", 
                      help="map method for output vector to density matrix (fac_t, fac_h, fac_a, proj_M, proj_S, proj_A)")
  parser.add_argument("--P_proj", type=float, default=1, help="coefficient for proj method")
  parser.add_argument("--r_path", type=str, default="results/result/")

  opt = parser.parse_args()

  # r_path = 'results/result/' + opt.na_state + '/'
  # results = Net_train(opt, device, r_path)


  # -----ex: 0 (Convergence Experiment of W State for Different Qubits, noise, limited measurements, LBFGS included)-----
  # set ty_state= 'pure', P_state = 1.0
  r_path = opt.r_path + 'QST/data/tetra_4/'
  for n_qubit in [2]:
      opt.n_qubits = n_qubit
      opt.n_samples = 100 * (opt.K ** opt.n_qubits)
      save_data = {}
      results = Net_train(opt, device, r_path)

      np.save(r_path +  str(opt.na_state) + '_' + str(n_qubit) + '_' + str(opt.P_state) + '.npy', results)


