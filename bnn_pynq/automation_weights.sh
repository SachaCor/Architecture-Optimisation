#!/bin/bash


filename="cfg/general.ini"


for i in {1..31}
do

first1=$((64+i-1))
first2=$((64+i))

last1=$((64-2*i+2))
last2=$((64-2*i))


var1="$first1,$last1,$first1"
var2="$first2,$last2,$first2"

sed -i -e 's/'$var1'/'$var2'/g' $filename

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 500 --experiments exp --gpus None

var=`ls exp -t | head -1`

BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/best.tar
done

sed -i -e 's/'$var2'/64,64,64/g' $filename
