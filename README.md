# **Efficient factored gradient descent algorithm for quantum state tomography**

The official Pytorch implementation of the paper named [`Efficient factored gradient descent algorithm for quantum state tomography`](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033034), has been accepted by Physical Review Research.

[![arXiv](https://img.shields.io/badge/arXiv-<2207.05341v4>-<COLOR>.svg)](https://arxiv.org/abs/2207.05341v4)

### **Abstract**

Reconstructing the state of quantum many-body systems is of fundamental importance in quantum information tasks, but extremely challenging due to the curse of dimensionality. In this work, we present an efficient quantum tomography protocol that combines the state-factored with eigenvalue mapping to address the rank-deficient issue and incorporates a momentum-accelerated gradient descent algorithm to speed up the optimization process. We implement extensive numerical experiments to demonstrate that our factored gradient descent algorithm efficiently mitigates the rank-deficient problem and admits orders of magnitude better tomography accuracy and faster convergence. We also find that our method can accomplish the full-state tomography of random 11-qubit mixed states within one minute.

### **Citation**

If you find our work useful in your research, please cite:

```
@article{PhysRevResearch.6.033034,
  title = {Efficient factored gradient descent algorithm for quantum state tomography},
  author = {Wang, Yong and Liu, Lijun and Cheng, Shuming and Li, Li and Chen, Jie},
  journal = {Phys. Rev. Res.},
  volume = {6},
  issue = {3},
  pages = {033034},
  numpages = {11},
  year = {2024},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.6.033034},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.6.033034}
}
```

## Getting started

This code was tested on the computer with a single Intel(R) Core(TM) i7-12700KF CPU @ 3.60GHz with 64GB RAM and a single NVIDIA GeForce RTX 3090 Ti GPU with 24.0GB RAM, and requires:

- Python 3.9
- conda3
- torch==2.0.1+cu118
- h5py==3.1.0
- matplotlib==3.5.2
- numpy==1.23.4
- openpyxl==3.0.10
- SciencePlots==2.0.1
- scipy==1.9.1
- tqdm==4.64.1

## Runs QST algorithms

```bash
python main.py
```

### 1. Initial Parameters (`main`)

```python
parser = argparse.ArgumentParser()
parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

parser.add_argument("--n_qubits", type=int, default=6, help="number of qubits")
parser.add_argument("--na_state", type=str, default="real_random_rank", help="name of state in library")
parser.add_argument("--P_state", type=float, default=0.1, help="P of mixed state")
parser.add_argument("--rank", type=float, default=2**5, help="rank of mixed state")
parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")

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
```

### 2. Run UGD with MRprop algorithm (`main`)

```python
print('\n'+'-'*20+'UGD_MRprop'+'-'*20)
gen_net = UGD_nn(opt.na_state, opt.n_qubits, P_idxs, M,
                                 map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

net = UGD(opt.na_state, opt.map_method, gen_net, data, opt.lr, optim_f="M")
result_save = {'parser': opt,
               'time': [],
               'epoch': [],
               'Fq': []}
net.train(opt.n_epochs, fid, result_save)
result_saves['UGD'] = result_save
```
### 3. Run UGD with MGD algorithm (`main`)

```python
print('\n'+'-'*20+'MGD'+'-'*20)
gen_net = UGD_nn(opt.na_state, opt.n_qubits, P_idxs, M,
                                 map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

net = UGD(opt.na_state, opt.map_method, gen_net, data, opt.lr, optim_f="S")
result_save = {'parser': opt,
               'time': [],
                'epoch': [],
                'Fq': []}
net.train(opt.n_epochs, fid, result_save)
result_saves['MGD'] = result_save
```
### 4. Run iMLE algorithm (`main`)

```python
print('\n'+'-'*20+'iMLE'+'-'*20)
result_save = {'parser': opt,
               'time': [],
                'epoch': [],
                'Fq': []}
iMLE(M, opt.n_qubits, data_all, opt.n_epochs, fid, result_save, device)
result_saves['iMLE'] = result_save
```

### 5. Run CG-APG algorithm (`main`)

```python
print('\n'+'-'*20+'CG-APG'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fq': []}
qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'proj_S', 2, result_save, device)
result_saves['APG'] = result_save
```

### 6. Run LRE algorithm (`main`)

```python
print('\n'+'-'*20+'LRE'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_S', 1, result_save, device)
result_saves['LRE'] = result_save
```

### 7. Run LRE algorithm with ProjA_1 (`main`)

```python
print('\n'+'-'*20+'LRE proj'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_A', 1, result_save, device)
result_saves['LRE_projA'] = result_save
```

#### **Acknowledgments**

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on: [POVM_GENMODEL](https://github.com/carrasqu/POVM_GENMODEL), [qMLE](https://github.com/qMLE/qMLE).

## **License**

This code is distributed under an [Mozilla Public License Version 2.0](LICENSE).

Note that our code depends on other libraries, including POVM_GENMODEL, qMLE, and uses algorithms that each have their own respective licenses that must also be followed.
