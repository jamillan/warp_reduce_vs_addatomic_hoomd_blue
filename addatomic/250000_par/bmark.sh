#!/bin/bash
#MSUB -N "test-bench"
#MSUB -l nodes=1:ppn=1:gpus=1
#MSUB -l walltime=48:00:00
#MSUB -A b1030
#MSUB -q buyin
#MSUB -l feature=k80

cd /projects/b1021/Jaime_pmf/test-wall/addatomic/250000_par


module load python
module load cuda/cuda_7.5.18_mpich

LD_LIBRARY_PATH=/projects/b1030/boost/lib:/software/anaconda2/lib:$LD_LIBRARY_PATH

python benchmark.py 

