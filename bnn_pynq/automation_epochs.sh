#!/bin/bash


BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 10 --scheduler FIXED --experiments exp --gpus None
var=`ls exp -t | head -1`
echo $var
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar


#First 500 with wide steps
for i in `seq 20 10 500`
do

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs $i --scheduler FIXED --experiments exp --gpus None --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar
done

#500 to 1000 with smaller steps
for i in `seq 505 5 1000`
do

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs $i --scheduler FIXED --experiments exp --gpus None --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar
done

#Last 1000 with wide steps
for i in `seq 1010 10 2000`
do

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs $i --scheduler FIXED --experiments exp --gpus None --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar

var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar
done

