#!/bin/bash
folder=test
python SAXS_Aniso_taichi.py -b 0.005 -q 1.2 -d $folder
echo "Computing average structure factor..."
python SAXS_Avg.py -d $folder
