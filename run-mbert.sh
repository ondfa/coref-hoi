#!/bin/bash
#PBS -N coref__mbert
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=25gb:scratch_local=10gb:ngpus=1:cl_zia=True
#PBS -l walltime=23:59:59
#PBS -j oe
#PBS -q gpu@cerit-pbs.cerit-sc.cz
#PBS -m e

cd /storage/plzen1/home/ondraprazak/coref-multiling-private
. /software/modules/init
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1
module add anaconda3-4.0.0
module add nccl/nccl-2.5.7-1-gcc-6.3.0-d7aro3u
module add openmpi-3.1.2-intel-cuda
source activate tensorflow-2-gpu
# python run.py train_czert_base 0
python run.py train_mbert_${lang} 0
# python evaluate.py train_mbert_slavic Jun05_16-32-22_87000 0
