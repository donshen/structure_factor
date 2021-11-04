#!/bin/bash
folder=239_460K_8x_2
python SAXS_Aniso_taichi.py -b 0.005 -q 1.2 -d $folder
echo "Computing average structure factor..."
python SAXS_Avg.py -d $folder
