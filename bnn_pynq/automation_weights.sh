#!/bin/bash

filename="cfg/general.ini"

#loop over the number of neurons to move
for i in {1..31}
do

#Add neurons to given layers
first1=$((64+i-1))
first2=$((64+i))

#Reduce neurons from a given layer
last1=$((64-2*i+2))
last2=$((64-2*i))

#Move neurons (last aims at the layer losing neurons, first aims at the layers recuperating neurons)
var1="$first1,$first1,$last1,"
var2="$first2,$first2,$last2"

#Move neurons
sed -i -e 's/'$var1'/'$var2'/g' $filename

#Train the network with the modified architecture
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 500 --experiments exp --gpus None

#Pick the last folder created
var=`ls exp -t | head -1`

#Test the network with the modified architecture
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume /home/cormenier/Documents/bnn_pynq_/exp/$var/checkpoints/best.tar
done

#Set the architecture to 64x64x64 at the end of the study
sed -i -e 's/'$var2'/64,64,64/g' $filename
