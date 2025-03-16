import torch
import numpy as np
from scipy import sparse
import scipy.linalg as sla
from scipy.linalg import norm
trace = lambda rho : np.real_if_close(np.trace(rho))
import ray


# === UTILITIES ===
I = np.eye(2)
X = np.array( [ [0,1],[1,0]] )
Y = np.array( [[ 0,-1j], [1j, 0] ] )
Z = np.array( [[1,0],[0,-1]] )
Pauli_map = { 0:I.reshape(1, -1), 1:Z.reshape(1, -1), 2:X.reshape(1, -1), 3:Y.reshape(1, -1) }
# dtype=np.single#np.int8 # np.int64, np.float64, etc., np.single, np.csingle, etc. Looks like if it's complex, has to be floating point not integer
# dtypeC = np.csingle
dtype,dtypeC = None, None  # let it be cast automatically
Xs = sparse.csr_matrix(X,dtype=dtype) # Todo: which is most efficient, csr or csc?
Ys = sparse.csr_matrix(Y,dtype=dtypeC) # also, use csr_matrix or csr_array?  csr_array is the future, but not all implemented yet
Zs = sparse.csr_matrix(Z,dtype=dtype)
Is = sparse.eye(2,format="csr",dtype=dtype)
# And some csc versions
Xcsc = sparse.csc_matrix(X,dtype=dtype)
Ycsc = sparse.csc_matrix(Y,dtype=dtypeC)
Zcsc = sparse.csc_matrix(Z,dtype=dtype)
Icsc = sparse.eye(2,format="csc",dtype=dtype)
# And some coo versions
Xcoo = sparse.coo_matrix(X,dtype=dtype)
Ycoo = sparse.coo_matrix(Y,dtype=dtypeC)
Zcoo = sparse.coo_matrix(Z,dtype=dtype)
Icoo = sparse.eye(2,format="coo",dtype=dtype)
# Think of I,Z,X,Y as 0,1,2,3
ind2Pauli_numeric = {0:Is, 1:Zs, 2:Xs, 3:Ys}
ind2Pauli_numeric_csc = {0:Icsc, 1:Zcsc, 2:Xcsc, 3:Ycsc}
ind2Pauli_numeric_coo = {0:Icoo, 1:Zcoo, 2:Xcoo, 3:Ycoo}
ind2Pauli_convert = {0:"I", 1:"Z", 2:"X", 3:"Y"}
ind2Pauli_letter  = {"I":Is, "Z":Zs, "X":Xs, "Y":Ys}

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

def explicit_Kronecker_Pauli( lst, nQubits = None, csc = False, coo = False ):
    """ builds up the explicit (sparse) matrix representing
    the Pauli matrix
    P = sigma_{i_1} \kron sigma_{i_2} \kron ... \kron sigma_{i_d}
     where sigma_{0 or 1 or 2 or 3} is a 2x2 Pauli matrix
     and list = [ i_1, i_2, ..., i_d ] is a list that represents
     which 2x2 Pauli matrices to use (d=# of qubits)

     Or, if 2nd input nQubits is supplied, then list can optionally be
      an integer in the range 0 <= ... < 4^nQubits, and will be converted
      to the corresponding Pauli via `int2lst`
    """
    if not isinstance( lst, list ):
        if nQubits is None:
            raise ValueError("Must supply nQubits as 2nd argument if first argument is an integer")
        lst = int2lst(lst,nQubits)

    if csc:
        indFcn = ind2Pauli_numeric_csc
    elif coo:
        indFcn = ind2Pauli_numeric_coo
    else:
        indFcn = ind2Pauli_numeric

    A = sparse.bsr_matrix([1])
    for i in lst:
        A = sparse.kron( A, indFcn[i] )
    return A

def pX(U):
    """ apply Pauli X operator, that is,
    U = [ [U1], [U2]]  -->  [ [U2], [U1] ]
    since sigma_X = [ [0,1],[1,0]]
    """
    return np.array( [U[1], U[0]] )

def pY(U):
    """ apply Pauli Y operator, that is,
    U = [ [U1], [U2]]  -->  [ -1j*[U2], 1j*[U1] ]
    since sigma_Y = [ [0,-1j],[1j,0]]
    """
    return np.array( [-1j*U[1], 1j*U[0]] )

def pZ(U):
    """ apply Pauli Z operator, that is,
    U = [ [U1], [U2]]  -->  [ [U1], -[U2] ]
    since sigma_Z = [ [1,0],[0,-1]]
    """
    return np.array( [U[0], -U[1]] )
def pI(U):
    """ identity operator """
    return U


PauliFcn = {0:pI, 1:pZ, 2:pX, 3:pY}

# PauliFcn = {0:pI, 1:pX, 2:pY, 3:pZ}





def pauliI(u,N,B):
    return u
def pauliX(u,N,B):
    # "B" for Block is what I called "dN" in my C code
    B2 = int(B/2)
    for i in range(0,N,B):
        temp = u[i:i + B2].copy() # don't know if we need the copy, but can't hurt...
        u[i:i + B2] = u[i+B2:i+B]
        u[i+B2:i+B] = temp
    return u
def pauliZ(u,N,B):
    B2 = int(B/2)
    for i in range(0,N,B):
        u[i+B2:i+B] *= -1
    return u
def pauliY(u,N,B):
    B2 = int(B/2)
    for i in range(0,N,B):
        temp = u[i:i + B2].copy() # don't know if we need the copy, but can't hurt...
        u[i:i + B2] = -1j*u[i+B2:i+B]
        u[i+B2:i+B] = 1j*temp
    return u
def lowmemAu_old(u,meas):
    """
    3/7/2025, borrowed from snippets of other codes
        uses just mList =[meas], doesn't loop

    y = lowmemA( U, meas)
        returns y = A( u )

        This is the building block for other codes, e.g., ones that do tr( A@u@u.T ) = vdot(u,A@u)
        or ones that build this up for a gradient
    """
    if u.ndim == 1: u = u.reshape( (-1,1) )
    r = u.shape[1]

    nQubits = int( np.log2( u.shape[0] ) )
    d = nQubits # alias

    Au = np.zeros((u.shape[0], u.shape[1]), dtype=np.complex64)

    # We'll be doing y_i = trace( A_i uu^* ) = trace( u^* A_i u)
    #   so then V u <-- A_i u, so  = trace( u^* v) = vdot(u,v)

    axs = np.arange(d)
    newAxs = np.roll( axs, 1 )
    if r==1:
        U = u.reshape( tuple( d*[2] ) ) # not sure if U and u are separate objectors or not...
        # todo: check np.may_share_memory(U,u) to see
    else:
        U = u.reshape( ( *tuple( d*[2] ), r ) )
        axs = np.hstack( [axs,d] ) # when u is a matrix with r > 1 columns
        newAxs = np.hstack( [newAxs,d] )


    V = U.copy()
    for p in int2lst(meas, nQubits ):
        # V = np.moveaxis( PauliFcn[p]( V ), axs, newAxs )
        #print(f'Using Pauli:  {ind2Pauli_convert[p]}')   # 2025 debugging
        V = PauliFcn[p]( V )   # PauliFcn = {0:pI, 1:pZ, 2:pX, 3:pY}
        #print( PauliFcn[p] )
        V = np.moveaxis( V, axs, newAxs )
    Au = V.reshape(-1, r)

    return Au

def implicit2explicit( linFcn, n):
    """ for now needs column vector inputs, and assumes square matrix
      This is used for debugging purposes
    """
    e = np.zeros( shape=(n,1) )
    At = np.zeros( shape=(n,n), dtype=np.complex128) # transpose, to make it efficient
    # collect output
    for i in range(n):
        e[i]=1
        At[i,:] = linFcn(e).T
        e[i]=0
    return np.real_if_close( At.T )

PauliFcn_vectorized = {0:pauliI, 1:pauliZ, 2:pauliX, 3:pauliY}

def lowmemAu_new(u,meas):
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
        B = 2**(ni+1) # block size
        # print(f'Qubit {ni}, using Pauli:  {ind2Pauli_convert[p]}, and block size {B}')
        u = PauliFcn_vectorized[p]( u, 2**nQubits, B)

    return u

def lowmemAu_old_gpu(u, meas_list):
    u = torch.tensor(u, dtype=torch.complex64, device='cuda')  # Move to GPU
    results = [lowmemAu_new(u, meas) for meas in meas_list]  # Use GPU tensors
    return torch.stack(results)  # Stack results along a new dimension

ray.init(num_gpus=4)

@ray.remote
def lowmemAu_distributed(u, meas):
    print(meas)
    return lowmemAu_new(u, meas)

def parallel_lowmemAu_ray(u, meas_list):
    futures = [lowmemAu_distributed.remote(u, m) for m in meas_list]
    return ray.get(futures)


d = 15 # number of qubits
nQubits = d # synonym
n = 2**d

r = 1 # rank
rng = np.random.default_rng(123)
u   = rng.standard_normal(size=(n,r)) + 1j*rng.standard_normal(size=(n,r))
u   = u / norm(u)
print(u.shape)
# rho = u@u.conj().T


meas = 1
lst  = int2lst(meas, nQubits )
print( lst )

print( "List of Paulis is", [ ind2Pauli_convert[i] for i in lst ] )

y  = lowmemAu_old( u, meas )
A2 = implicit2explicit( lambda u : lowmemAu_old( u, meas ), n )

# A  = explicit_Kronecker_Pauli( lst, nQubits )
# y2 = A @ u

# print(f'Discrepancy between two OLD versions of code is is {norm(y-y2)/norm(y)}')

# y3 = lowmemAu_new(u,meas)
# print(f'Discrepancy between old and new versions of code is is {norm(y-y3)/norm(y)}')

y3 = parallel_lowmemAu_ray(u, np.arange(4**d))
print(f'Discrepancy between old and new versions of code is is {norm(y-y3[1])/norm(y)}')
print(y3.shape)
