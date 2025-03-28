import pickle 
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_fval_vs_qubits(dir, save_destination=None):
    output = {}
    #get file names
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.npy'): 
            data = np.load(os.path.join(dir, filename), allow_pickle=True)
            data = data.item()
            args = (data['UGD']['parser'])
            n_qubits = args.n_qubits
            n_samples = args.n_samples
            if n_qubits not in output:
                output[n_qubits] = {}
            fval_min = np.inf
            for algo in data.keys():
                fval = data[algo]['loss']
                if len(fval) != 0:
                  fval_min = torch.minimum(torch.tensor(fval_min), torch.min(torch.tensor(fval)))


            for algo in data.keys():

              fval = data[algo]['loss']
              condition = (torch.tensor(fval)-fval_min) <= 1e-3 # -1/n_samples * np.log(0.95)
              indices = torch.where(condition)[0]
              if len(indices) > 0:
                idx = indices[0].item()  # Get first occurrence of True
                time_taken = data[algo]['time'][idx]
              else:
                time_taken = float('inf')
      
              output[n_qubits][algo] = time_taken


    res = {}
    algos = list(output[8].keys())
    for algo in algos:
      # print(algo.dtype)
      if algo == 'lbfgs':
        algo = 'L-BFGS'
      elif algo == 'APG':
        algo = 'CG-APG'   
      
      res.setdefault(algo, {}) 
      
    # n_qubits = list(output.keys())
    
    for algo in algos:
      if algo == 'lbfgs':
        algo_ = 'L-BFGS'
      elif algo == 'APG':
        algo_ = 'CG-APG'
      else:
        algo_ = algo  
      
      for key in output.keys():
         
        res[algo_][key] = output[key][algo]

    res.pop('LRE')
    res.pop('LRE_projA')    
    
    print(res)
    #Plot the data
    markers = {'L-BFGS':'v', 'CG-APG':'^', 'UGD': 'o', 'MGD': 'x', 'iMLE':'d', 'LRE': 's', 'LRE_projA':'p'}
    colors = {'L-BFGS':'#8c564b', 'CG-APG':'#9467bd', 'UGD': '#1f77b4', 'MGD': '#ff7f0e', 'iMLE':'#2ca02c', 'LRE': '#d62728', 'LRE_projA':'#e377c2'}
    fontSize=12
    plt.rcParams.update({
    'font.size': fontSize,         # Set font size for labels, legends, and ticks
    'axes.labelsize': fontSize,    # X and Y labels
    'legend.fontsize': fontSize,   # Legend
    'xtick.labelsize': fontSize,   # X-axis tick labels
    'ytick.labelsize': fontSize    # Y-axis tick labels
    })
    # plt.figure(figsize=(10, 6))

    for algo, results in res.items():
      if algo != 'LRE' or 'LRE_projA':
          # Sort inner dictionary by key
          sorted_items = sorted(results.items()) 

          x, y = zip(*[(k, v) for k, v in sorted_items if k != 7])  # Filter out inf values
          
          # if x:  # Only plot if there are valid values
          ax = plt.semilogy(x, y, color = colors[algo] ,  marker=markers[algo], markerfacecolor='none', label=algo)
          
          

    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (s)')
    plt.grid(True, which='both', linestyle='--', alpha = 1.0 , linewidth = 0.3, dashes=(2, 10))
    plt.title("Algorithm Performance Comparison")
    plt.legend(loc = 'upper left')
    plt.show()

    # if save_destination != None:
    #     plt.savefig(save_destination+'/fval_vs_qubit_maxcor_2_high_acc_new.pdf', format='pdf')
    # plt.show()     
    
