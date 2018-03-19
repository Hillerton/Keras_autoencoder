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

nodes=64
epoch=100
infile=/home/hillerton/Data/cancer_filtered_bed/
outdir=/home/hillerton/results #/home/t/thohi921/pfs/test_data"/"$DATE"_ae_keras_run"
logfile=$outdir"/"$DATE"_autoencoder_run"
model=/home/hillerton/results/2018-03-18_ae_test/AE_model.h5 #/home/hillerton/results/2018-03-15ae_keras_run_1000g_all/2018-03-15_ae_test/AE_model.h5
subset=50000
pheno=/home/hillerton/Data//exome_phenotype.txt #/home/hillerton/Data/cancer_patient_exome_seq/exome_phenotype.txt

mkdir -p $outdir >>/dev/null

if [ ! -f $logfile ];
then
  touch $logfile
fi

#python3 init_autoencoder.py $infile $nodes $epoch --out $outdir --log $logfile --subset $subset  --model $model --regression True --phenotypes $pheno #local
python3 init_autoencoder.py $infile $nodes $epoch --out $outdir --log $logfile --model $model --regression True --phenotypes $pheno #cluster based
