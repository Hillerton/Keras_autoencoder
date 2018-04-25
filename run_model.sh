#!/bin/bash

#SBATCH -A SNIC2017-1-526
#SBATCH -J thohi_autoencoder_test
#SBATCH --time=02:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:k80:1

#source /home/t/thohi921/pfs/keras_vrtenv/bin/activate

#load modules
#module add GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
#module add TensorFlow/1.5.0-Python-3.6.3
#module add Keras/2.1.2-Python-3.6.3


#export PYTHONPATH=$PYTHONPATH:/home/t/thohi921/pfs/keras_vrtenv/lib/python3.6/site-packages

DATE=`date +%Y-%m-%d`

nodes=128
epoch=100
train_files=/home/hillerton/Data/intersect_1000g_cancer/1000_genome/bed_files/chr6
test_files=/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files
outdir=/home/hillerton/results
logfile=$outdir"/"$DATE"_autoencoder_run"
noise=0.001


mkdir -p $outdir >>/dev/null

if [ ! -f $logfile ];
then
  touch $logfile
fi

python3 run_autoencoder.py $train_files $test_files $nodes $epoch --out $outdir --log $logfile --noise $noise #--subset 5000
