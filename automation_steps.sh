#!/bin/bash


#BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 2 --experiments exp --gpus None


for i in `seq 100 100 2000`
do

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 10000 --experiments exp --gpus None --scheduler FIXED --stepLR $i


var=`ls exp -t | head -1`

echo $var

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar --noiseT 0
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar --noiseT 0.01
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar --noiseT 0.02
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar --noiseT 0.05
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/checkpoint.tar --noiseT 0.1
done

