from lowmem_main import *
import heapq
import numpy as np
impprt scipy

def get_proposal_pmf(u, n):
  K = np.arange(4)
  for i in reversed(range(n)):
    p = p_x(K*4**(i), u)
    p = np.maximum(p, 1e-36)
    p /= p.sum()
    p = np.log(p)
    p = p.astype(np.float64)

    pmf.append(p)
  return pmf


def p_x(idx, u):

  if u.ndim == 1: u = u.reshape( (-1,1) ) # make sure it's a column vector
  
  nQubits = int( np.log2( u.shape[0] ) )
  
  y = np.zeros((idx.shape[0]))
  k = 0
  for i in idx:
    v = u.copy()
    for ni,p in enumerate( reversed(int2lst(i, nQubits )) ):
      v = PauliFcn_map[p]( v, 2**nQubits, ni)
    y[k] = np.real_if_close(np.vdot(u.copy(), v)**2/2**nQubits)
    k += 1
  
  return y


def heap_sort(pmf_list, K):
# Step 1: Sort each vector descending
sorted_indices = []
sorted_values = []

for pmf in pmf_list:
  idx = np.argsort(-pmf)  # descending
  sorted_indices.append(idx)
  sorted_values.append(pmf[idx])

# Step 2: Initialize heap
n = len(pmf_list)

# Max heap: store (-sum of logs, indices)
initial_log_sum = sum([v[0] for v in sorted_values])
heap = [(-initial_log_sum, [0]*n)]  # negative because heapq is a min-heap
heapq.heapify(heap)

# Bookkeeping
seen = set()
results = []

while len(results) < K:
    neg_prod, indices = heapq.heappop(heap)
    results.append(indices.copy())
    
    for dim in range(n):
        new_indices = indices.copy()
        if new_indices[dim] + 1 < 4:
            new_indices[dim] += 1
            new_key = tuple(new_indices)
            if new_key not in seen:
                # Inserted check
                # if sorted_values[dim][new_indices[dim]] != 0:
                new_log_sum = sum([
                sorted_values[d][new_indices[d]] for d in range(n)])
                heapq.heappush(heap, (-new_log_sum, new_indices))
                seen.add(new_key)

return results


