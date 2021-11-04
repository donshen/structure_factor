import os
import time
import argparse
import itertools
import numpy as np
from numba import njit, prange
from tqdm import tqdm
from timeit import default_timer as timer
start = time.perf_counter()

parser = argparse.ArgumentParser(description='Compute X-ray scattering structure factor for anisotropic systems')

parser.add_argument('-q','--qmax',
                    help ='max q in A^-1',
                    type=float)

parser.add_argument('-b','--bin_size',
                    help ='bin width in A',
                    type=float)

@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res

@njit(parallel=True)
def mat_cos(A):
    res = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            res[i,j] += np.cos(A[i,j])
    return res

@njit(parallel=True)
def mat_sin(A):
    res = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            res[i,j] += np.sin(A[i,j])
    return res

@njit(parallel=True)
def mat_pow(A, n):
    res = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            res[i,j] += A[i,j] ** n
    return res

@njit
def mat_reshape_N_1(A):
#     assert len(A.shape) < 2
    return A.reshape(A.shape[0], -1)
    

@njit(parallel=True)
def mat_elem_mult(A, B):
    A, B = mat_reshape_N_1(A), mat_reshape_N_1(B)
    assert A.shape[0] == B.shape[0]
    res = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            if A.shape[1] == B.shape[1]:
                res[i,j] += A[i,j] * B[i, j]
            else:
                assert B.shape[-1] == 1
                res[i,j] += A[i,j] * B[i, 0]
    return res

def form_factor(q_abs, atom_type):
    ff_a = {'O': np.array([[3.0485, 2.2868, 1.5463, 0.867]]),
            'H': np.array([[0.489918, 0.262003, 0.196767, 0.049879]]),
            'C': np.array([[2.31, 1.02, 1.5886, 0.865]]),
           }
    ff_b = {'O': np.array([[13.2771, 5.7011, 0.3239, 32.9089]]),
            'H': np.array([[20.6593, 7.74039, 49.5519, 2.20159]]),
            'C': np.array([[20.8439, 10.2075, 0.5687, 51.6512]]),
           }
    ff_c = {'O': 0.2508, 'H': 0.001305, 'C': 0.2156}

    try:
#         return np.sum(ff_a[atom] * np.exp(-ff_b[atom]*(q_abs/4/np.pi)**2)) + ff_c[atom]
        q_abs = q_abs.reshape((1, q_abs.shape[0]))
        term1 = ff_a[atom_type]
        term2 = np.exp(np.matmul(((q_abs / 4 / np.pi) ** 2).T, -ff_b[atom_type]))
        return np.dot(term2, term1.T) + ff_c[atom_type]
    except KeyError:
        raise ValueError("unknown atom type!")
        
def get_ft_factors(q_abs, q_vec, coords, atom_type):
    coords_atom = coords[atom_list == atom_type, :]    
    fq_atom = form_factor(q_abs, atom_type)
    dot_prod_atom = mat_mult(q_vec, coords_atom.T)
    cos_atom = np.sum(mat_elem_mult(mat_cos(dot_prod_atom), fq_atom), axis=1)
    sin_atom = np.sum(mat_elem_mult(mat_sin(dot_prod_atom), fq_atom), axis=1)
    return cos_atom, sin_atom  

if __name__ == "__main__":
    args = parser.parse_args()
    q_max = args.qmax

    pdb_files = [f for f in os.listdir('./test') if f[-4:] == '.pdb']

    for file in tqdm(pdb_files, desc=f"Computing structure factors for each frame..."):
        read_file = open(os.path.join('./test', file), 'r')
        atom_list, coords = [], []        

        # Read input
        for line in read_file:
            # Read box size
            if line.split()[0] == 'CRYST1':
                line = line.split()
                L = np.array(line[1:4], dtype=np.float32)

            # Read atom coordinates
            elif line.split()[0] == 'ATOM': 
                line = line.split()
                coord = np.array(line[-6:-3], dtype=np.float32)
                atom = line[-1]
                atom_list.append(atom[0])
                coords.append(coord)
        coords = np.vstack(coords)
        read_file.close()
        
        # Compute scattering

        q_min = 2 * np.pi / L
        ndx_max = (q_max / q_min).astype(int) + 1
        q_bin_width = args.bin_size
        q_bin_num = int(q_max / q_bin_width)
        cos, sin = 0, 0
        atom_list = np.array(atom_list)
        q_vectors, q_list, Sq_list, ndx_list = [], [], [], []
        Sq_avg_list = [0] * q_bin_num
        bin_num_list = [0] * q_bin_num
        fq_O, fq_H, fq_C = [], [], []

        # Iterate through the bins in x, y, z directions
        ind_x, ind_y, ind_z = np.arange(-ndx_max[0], ndx_max[0]), np.arange(-ndx_max[1], ndx_max[1]), np.arange(-ndx_max[2], ndx_max[2])
        ndx_list = list(itertools.product(ind_x, ind_y, ind_z))

        q_vec = np.array(ndx_list) * q_min
        q_abs = np.einsum('ij,ij->i', q_vec, q_vec) ** 0.5
        cond = np.logical_and(q_abs > 0, q_abs < q_max)
        q_vec, q_abs, ndx_list = q_vec[cond], q_abs[cond], np.array(ndx_list)[cond]
        ndx_list_str = [f'{ndx[0]}-{ndx[1]}-{ndx[2]}' for  ndx in ndx_list]

        cos_O, sin_O = get_ft_factors(q_abs, q_vec, coords, 'O')
        cos_H, sin_H = get_ft_factors(q_abs, q_vec, coords, 'H')
        cos_C, sin_C = get_ft_factors(q_abs, q_vec, coords, 'C')  
        cos = cos_O + cos_H + cos_C
        sin = sin_O + sin_H + sin_C
        Sq_list = mat_elem_mult(cos, cos) + mat_elem_mult(sin, sin)
        fq_squaresum = np.sum([mat_pow(form_factor(q_abs, atom), 2) for atom in ['O', 'H', 'C']])

        for q, Sq in zip(q_abs, Sq_list):
            bin_ndx = int(q / q_bin_width)
            Sq_avg_list[bin_ndx] += Sq
            bin_num_list[bin_ndx] += 1

        # Output results
        output = open(os.path.join('./test/', f'Sq-{file}.dat'), 'w')
        output.write('# q [A^-1] Sq ndx_x-ndx_y-ndx_z\n')
        for i in range(len(q_abs)):
            output.write('%7.4f %8.3f %s\n' % (q_abs[i], Sq_list[i], ndx_list_str[i]))
        output.close()

        # Output averaged results
        output = open(os.path.join('./test', f'Sq-avg-{file}.dat'), 'w')
        output.write('# q [A^-1] Sq\n')
        for i in range(len(bin_num_list)):
            if bin_num_list[i] > 0:
                Sq_avg_out = Sq_avg_list[i] / bin_num_list[i]
            else:
                Sq_avg_out = 0
            output.write('%7.4f %8.3f\n' % (q_bin_width * (i+0.5), Sq_avg_out/fq_squaresum))
        output.close()
