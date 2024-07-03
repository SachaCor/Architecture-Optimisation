#Import libraries

import pickle
import torch
import sys
import os

#Open file

path='exp/'+str(sys.argv[1])+'/checkpoints/'
if os.path.isfile(path+'best.tar'):
	package = torch.load(path+'best.tar', map_location='cpu')
else:
	package = torch.load(path+'checkpoint.tar', map_location='cpu')
a=package['state_dict']
b=package['optim_dict']['state']

ini_0=torch.tensor([0])
ini_1=torch.tensor([1])

#WEIGHTS

#Weights Input/1

temp3=torch.normal(0, 1, size=(1, 784))
a['features.2.weight'] = torch.cat((a['features.2.weight'],temp3),0)

print(len(a['features.2.weight'][64]))
#Weights 1/2

NumberOfNeuronsFirst1=len(a['features.7.weight'])

	#Add 1 weight for all the neurons on the second layer

temptens=torch.tensor([])
for k in range (0,NumberOfNeuronsFirst1):
	truc=torch.normal(0, 1, size=(1,1))[0]
	temp=torch.cat((a['features.6.weight'][k],truc),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
a['features.6.weight']=temptens

	#Add the moving neuron weights

temp3=torch.normal(0, 1, size=(1, NumberOfNeuronsFirst1+1))
a['features.6.weight'] = torch.cat((a['features.6.weight'],temp3),0)

#Weights 2/3

	#Delete the last neurons

a['features.10.weight']=a['features.10.weight'][0:len(a['features.10.weight'])-2]

	#Add 1 weight for all the neurons in the third layer

temptens=torch.tensor([])
for k in range (0,len(a['features.10.weight'])):
	truc=torch.normal(0, 1, size=(1,1))[0]
	temp=torch.cat((a['features.10.weight'][k],truc),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
a['features.10.weight']=temptens
print(len(a['features.10.weight']))

#Weights 3/Output

temptens=torch.tensor([])
for k in range (0,10):
	truc=torch.normal(0, 1, size=(1,1))[0]
	temp=a['features.14.weight'][k][0:len(a['features.14.weight'][k])-2]
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
a['features.14.weight']=temptens
#WeightNeuron

NumberOfNeuronsLast1=len(a['features.11.weight'])
temp1=a['features.11.weight'][NumberOfNeuronsLast1-1]
a['features.7.weight'] = torch.cat((a['features.7.weight'],ini_0),0)
a['features.11.weight']=a['features.11.weight'][0:len(a['features.11.weight'])-1]


NumberOfNeuronsLast2=len(a['features.11.weight'])
temp2=a['features.11.weight'][NumberOfNeuronsLast2-1]
a['features.3.weight'] = torch.cat((a['features.3.weight'],ini_0),0)
a['features.11.weight']=a['features.11.weight'][0:len(a['features.11.weight'])-1]

#Bias

temp1=a['features.11.bias'][NumberOfNeuronsLast1-1]
a['features.7.bias'] = torch.cat((a['features.7.bias'],ini_0),0)
a['features.11.bias']=a['features.11.bias'][0:len(a['features.11.bias'])-1]

temp2=a['features.11.bias'][NumberOfNeuronsLast2-1]
a['features.3.bias'] = torch.cat((a['features.3.bias'],ini_0),0)
a['features.11.bias']=a['features.11.bias'][0:len(a['features.11.bias'])-1]

#Running_mean

temp1=a['features.11.running_mean'][NumberOfNeuronsLast1-1]
a['features.7.running_mean'] = torch.cat((a['features.7.running_mean'],ini_0),0)
a['features.11.running_mean']=a['features.11.running_mean'][0:len(a['features.11.running_mean'])-1]

temp2=a['features.11.running_mean'][NumberOfNeuronsLast2-1]
a['features.3.running_mean'] = torch.cat((a['features.3.running_mean'],ini_0),0)
a['features.11.running_mean']=a['features.11.running_mean'][0:len(a['features.11.running_mean'])-1]

#Running_var

temp1=a['features.11.running_var'][NumberOfNeuronsLast1-1]
a['features.7.running_var'] = torch.cat((a['features.7.running_var'],ini_0),0)
a['features.11.running_var']=a['features.11.running_var'][0:len(a['features.11.running_var'])-1]

temp2=a['features.11.running_var'][NumberOfNeuronsLast2-1]
a['features.3.running_var'] = torch.cat((a['features.3.running_var'],ini_0),0)
a['features.11.running_var']=a['features.11.running_var'][0:len(a['features.11.running_var'])-1]

#Opt

idx=0
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],torch.zeros([1,len(b[idx]['exp_avg'][0])])),0)
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],torch.zeros([1,len(b[idx]['exp_avg_sq'][0])])),0)
idx=1
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],ini_0),0)
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],ini_0),0)
idx=2
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],ini_0),0)
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],ini_0),0)
idx=3
temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg'])):
	temp=torch.cat((b[idx]['exp_avg'][k],ini_0),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg']=temptens
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],torch.zeros([1,len(b[idx]['exp_avg'][0])])),0)

temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg_sq'])):
	temp=torch.cat((b[idx]['exp_avg_sq'][k],ini_0),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg_sq']=temptens
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],torch.zeros([1,len(b[idx]['exp_avg_sq'][0])])),0)
idx=4
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],ini_0),0)
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],ini_0),0)
idx=5
b[idx]['exp_avg']=torch.cat((b[idx]['exp_avg'],ini_0),0)
b[idx]['exp_avg_sq']=torch.cat((b[idx]['exp_avg_sq'],ini_0),0)
idx=6
b[idx]['exp_avg']=b[idx]['exp_avg'][0:len(b[idx]['exp_avg'])-2]
temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg'])):
	temp=torch.cat((b[idx]['exp_avg'][k],ini_0),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg']=temptens

b[idx]['exp_avg_sq']=b[idx]['exp_avg_sq'][0:len(b[idx]['exp_avg_sq'])-2]
temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg_sq'])):
	temp=torch.cat((b[idx]['exp_avg_sq'][k],ini_0),0)
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg_sq']=temptens
idx=7
b[idx]['exp_avg']=b[idx]['exp_avg'][0:len(b[idx]['exp_avg'])-2]
b[idx]['exp_avg_sq']=b[idx]['exp_avg_sq'][0:len(b[idx]['exp_avg_sq'])-2]
idx=8
b[idx]['exp_avg']=b[idx]['exp_avg'][0:len(b[idx]['exp_avg'])-2]
b[idx]['exp_avg_sq']=b[idx]['exp_avg_sq'][0:len(b[idx]['exp_avg_sq'])-2]
idx=9
temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg'])):
	temp=b[idx]['exp_avg'][k][0:len(b[idx]['exp_avg'][k])-2]
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg']=temptens

temptens=torch.tensor([])
for k in range (0,len(b[idx]['exp_avg_sq'])):
	temp=b[idx]['exp_avg'][k][0:len(b[idx]['exp_avg_sq'][k])-2]
	temptens=torch.cat((temptens,temp.unsqueeze(0)),0)
b[idx]['exp_avg_sq']=temptens

#Save new file

torch.save({
            'state_dict': a,
            'optim_dict': package['optim_dict'],
            'epoch': 1,
            'best_val_acc': package['best_val_acc'],
        }, path+'best2.tar')

#Check results
#print(a.keys())
#print(type(a['features.2.weight'][784]))
#print(temp3)
#print(a['features.7.weight'])
#print(a['features.11.weight'])
#print(temp2)
