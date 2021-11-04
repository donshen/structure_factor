import os
import time
import argparse
import itertools
import numpy as np
import taichi as ti
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compute X-ray scattering structure factor for anisotropic systems')

parser.add_argument('-q','--qmax',
                    help='max q in A^-1',
                    type=float)

parser.add_argument('-b','--bin_size',
                    help='bin width in A',
                    type=float)
parser.add_argument('-d','--data_path',
                    help='the path to read pdb files and save output data',
                    type=str)

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
        q_abs = q_abs.reshape((1, q_abs.shape[0]))
        term1 = ff_a[atom_type]
        term2 = np.exp(np.matmul(((q_abs / 4 / np.pi) ** 2).T, -ff_b[atom_type]))
        return np.dot(term2, term1.T) + ff_c[atom_type]
    except KeyError:
        raise ValueError("unknown atom type!")

def form_factor_coeffs(atom_type):
    ff_a = {'O': np.array([3.0485, 2.2868, 1.5463, 0.867], dtype=np.float32),
            'H': np.array([0.489918, 0.262003, 0.196767, 0.049879], dtype=np.float32),
            'C': np.array([2.31, 1.02, 1.5886, 0.865], dtype=np.float32),
           }
    ff_b = {'O': np.array([13.2771, 5.7011, 0.3239, 32.9089], dtype=np.float32),
            'H': np.array([20.6593, 7.74039, 49.5519, 2.20159], dtype=np.float32),
            'C': np.array([20.8439, 10.2075, 0.5687, 51.6512], dtype=np.float32),
           }
    ff_c = {'O': 0.2508, 'H': 0.001305, 'C': 0.2156}

    return ff_a[atom_type], ff_b[atom_type], ff_c[atom_type]

def get_ft_factors(q_abs, q_vec, coords, atom_type):
    coords_atom = coords[atom_list == atom_type, :]
    cos_atom = np.empty_like(q_abs, dtype=np.float32)
    sin_atom = np.empty_like(q_abs, dtype=np.float32)
    get_ft_factors_kernel(
        q_abs.astype(np.float32),
        q_vec.astype(np.float32),
        coords_atom.astype(np.float32),
        *form_factor_coeffs(atom_type),
        cos_atom,
        sin_atom,
    )
    return cos_atom, sin_atom


@ti.kernel  
def get_ft_factors_kernel(
    q_abs: ti.any_arr(),
    q_vec: ti.any_arr(),
    coords: ti.any_arr(),
    ff_a: ti.any_arr(),
    ff_b: ti.any_arr(),
    ff_c: ti.f32,
    cos_atom: ti.any_arr(),
    sin_atom: ti.any_arr(),
):
    # automatically parallelize the top-level for loop
    for i in range(q_abs.shape[0]): 
        # calculate form factor
        form_factor = ff_c
        for k in ti.static(range(4)):
            form_factor += ff_a[k] * ti.exp(-ff_b[k] * (q_abs[i] / 4 / np.pi) ** 2)
        # calculate ft factor
        cos_atom[i] = 0.0
        sin_atom[i] = 0.0
        for j in range(coords.shape[0]):
            dot_prod_atom = q_vec[i, 0] * coords[j, 0] \
                + q_vec[i, 1] * coords[j, 1] \
                + q_vec[i, 2] * coords[j, 2]
            cos_atom[i] = cos_atom[i] + form_factor * ti.cos(dot_prod_atom)
            sin_atom[i] = sin_atom[i] + form_factor * ti.sin(dot_prod_atom)


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    args = parser.parse_args()
    q_max = args.qmax

    pdb_files = [f for f in os.listdir(args.data_path) if f[-4:] == '.pdb']

    for file in tqdm(pdb_files, desc=f"Computing structure factors for each frame..."):
        read_file = open(os.path.join(args.data_path, file), 'r')
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
        Sq_list = cos * cos + sin * sin
        fq_squaresum = np.sum([form_factor(q_abs, atom) ** 2 for atom in ['O', 'H', 'C']])
        for q, Sq in zip(q_abs, Sq_list):
            bin_ndx = int(q / q_bin_width)
            Sq_avg_list[bin_ndx] += Sq
            bin_num_list[bin_ndx] += 1
        # Output results
        output = open(os.path.join(args.data_path, f'Sq-{file}.dat'), 'w')
        output.write('# q [A^-1] Sq ndx_x-ndx_y-ndx_z\n')
        for i in range(len(q_abs)):
            output.write('%7.4f %8.3f %s\n' % (q_abs[i], Sq_list[i], ndx_list_str[i]))
        output.close()
        # Output averaged results
        output = open(os.path.join(args.data_path, f'Sq-avg-{file}.dat'), 'w')
        output.write('# q [A^-1] Sq\n')
        for i in range(len(bin_num_list)):
            if bin_num_list[i] > 0:
                Sq_avg_out = Sq_avg_list[i] / bin_num_list[i]
            else:
                Sq_avg_out = 0
            output.write('%7.4f %8.3f\n' % (q_bin_width * (i+0.5), Sq_avg_out/fq_squaresum))
        output.close()
