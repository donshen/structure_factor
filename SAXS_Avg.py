# Average the scattering intensity for every q and compute the uncertainty

import numpy as np
import os
import scipy as sp
import scipy.stats
import argparse

parser = argparse.ArgumentParser(description='Compute average structure factors')
parser.add_argument('-d','--data_path',
                    help='the path to read pdb files and save output data',
                    type=str)
args = parser.parse_args() 

confidence = 0.95

i = 0
for filename in os.listdir(args.data_path):
    if filename.startswith("Sq-avg") and filename.endswith(".dat") and not filename.endswith("avg.dat"): 
        if i == 0:
            data = np.loadtxt(os.path.join(args.data_path, filename)) 
            i += 1
        else:
            data_temp = np.loadtxt(os.path.join(args.data_path, filename))
            data = np.append(data, data_temp[:,1][...,None], 1) 
    else:
        continue

Sq = data[:,1:len(data[0,:])]
q_out = []
Sq_mean = []
Sq_h = []

for i in range(len(data[:,0])):
    Sq_nonzero = [x for x in Sq[i,:] if x > 0]
    n = len(Sq_nonzero)

    if n > 1:
        mean = np.mean(Sq_nonzero)
        sem = scipy.stats.sem(Sq_nonzero)
        h = sem * sp.stats.t._ppf((1+confidence)/2., n-1)
        q_out.append(data[i,0])
        Sq_mean.append(mean)
        Sq_h.append(h)

Output = np.column_stack((q_out, Sq_mean, Sq_h))
np.savetxt(os.path.join(args.data_path, 'Sq_avg_aniso.dat'), Output, fmt = '%6.4f %7.3f %7.3f')
