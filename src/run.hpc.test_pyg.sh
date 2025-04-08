#!/bin/bash
#JSUB -J test_singularity_gpu
#JSUB -q gpu
#JSUB -n 1
#JSUB -gpgpu '1 mig=1'
#JSUB -o log_out.%J
#JSUB -e log_err.%J

module purge
module load singularity-4.2.1

MY_HOME=/storage/public/home/2022051346

DEEPTAN_HOME=$MY_HOME/prj/deeptan
SIF=$DEEPTAN_HOME/deeptan.sif

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec -B $DEEPTAN_HOME:$DEEPTAN_HOME --nv $SIF python -c "import torch; print(torch.cuda.is_available())"
