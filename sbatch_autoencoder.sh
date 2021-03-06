#!/bin/bash

<<coment
script to feed parameters to init_autoencoder when ran locally or on cluster 
coment

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

echo "begining"

DATE=`date +%Y-%m-%d`

nodes=254
epoch=100
infile=/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files
outdir=/home/hillerton/results/2018-05-03_ae_keras_run/
logfile=$outdir"/"$DATE"_autoencoder_run"
model=/home/hillerton/results/2018-05-03_ae_keras_run/AE_model.h5
pheno=/home/hillerton/Data/cancer_patient_exome_seq/exome_phenotype.txt
noise=0.001

mkdir -p $outdir >>/dev/null

if [ ! -f $logfile ];
then
  touch $logfile
fi

#python3 init_autoencoder.py $infile $nodes $epoch --out $outdir --log $logfile --subset $subset  --model $model --regression True --phenotypes $pheno #local


for i in ${nodes[@]};
do
    python3 init_autoencoder.py $infile $i $epoch --out $outdir --log $logfile --model $model --phenotypes $pheno --noise $noise --regression True
done
