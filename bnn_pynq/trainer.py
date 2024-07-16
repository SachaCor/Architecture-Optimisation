# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import os
import time
from datetime import datetime
import cv2
import glob
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from logger import Logger, TrainingEpochMeters, EvalEpochMeters
from models import model_with_cfg
from models.losses import SqrHingeLoss

torch.set_printoptions(profile="full")

#Function to add random noise in the images
class AddRandomNoise(object):		
    
    def __init__(self,fraction):
        self.fraction = fraction

    def __call__(self,nparray):
        flip_mask = np.random.rand(28,28) < self.fraction
        flip_mask = torch.from_numpy(flip_mask)
        noisy_data=(1 - (flip_mask*1)) * nparray + flip_mask * (1 - nparray)
        return noisy_data

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def Average(lst):
   return sum(lst)/len(lst)


class Trainer(object):
    def __init__(self, args):
        
        #Set the seed for the randomness
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        
        model, cfg = model_with_cfg(args.network, args.pretrained)

        #Init arguments
        self.args = args
        prec_name = "_{}W{}A".format(cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH'),
                                     cfg.getint('QUANT', 'ACT_BIT_WIDTH'))
        experiment_name = '{}{}_{}'.format(args.network, prec_name,
                                           datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir_path = os.path.join(args.experiments, experiment_name)
        
        if self.args.evaluate:
            self.output_dir_path, _ = os.path.split(args.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

        if not args.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            if not args.evaluate:
            	os.mkdir(self.output_dir_path)
            	os.mkdir(self.checkpoints_dir_path)
        self.logger = Logger(self.output_dir_path, args.dry_run)

        # Datasets
        NoiseRateT=self.args.noiseT		#Choose the percentage of noise
        dataset = cfg.get('MODEL', 'DATASET')
        self.num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
        transform_train = transforms.Compose([transforms.ToTensor(),AddRandomNoise(NoiseRateT)])
        transform_test = transforms.Compose([transforms.ToTensor(),AddRandomNoise(NoiseRateT)])

   ####################################################
        dataset=self.args.dataset
        #Create Train, Valid and Test sets
        train_data_path = 'images/'+dataset+'/train'
        test_data_path = 'images/'+dataset+'/test'

        train_image_paths = [] #to store image paths in list
        classes = [] #to store class values
        #Choose the path for training and testing datasets
        path_train = []
        path_test = []
        for class_name in sorted(os.listdir(train_data_path)):
            path_train.append([train_data_path+'/'+class_name])
            path_test.append([test_data_path+'/'+class_name])
            # get all the paths from train_data_path and append image paths and class to respective lists    
            # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
        for data_path in path_train: 
            print('train_image_path example: ',data_path[0] )            
            classes.append(data_path[0].split('/')[-1]) 
            train_image_paths.append(glob.glob(data_path[0] + '/*'))

       
        train_image_paths = list(np.concatenate(train_image_paths).flat)
        random.shuffle(train_image_paths)

        print('class example: ', classes)
 
        test_image_paths = []
        for data_path in path_test:
            test_image_paths.append(glob.glob(data_path[0] + '/*'))

        test_image_paths = list(np.concatenate(test_image_paths).flat)
        self.test = test_image_paths



  #######################################################
  #      Create dictionary for class indexes
  #######################################################

        idx_to_class = {i:j for i, j in enumerate(classes)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}

        class MDataset(Dataset):
            def __init__(self, image_paths, transform=False):
               self.image_paths = image_paths
               self.transform = transform
        
            def __len__(self):
               return len(self.image_paths)

            def __getitem__(self, idx):
                image_filepath = self.image_paths[idx]
                image = cv2.imread(image_filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
                label = image_filepath.split('/')[-2]
                label = class_to_idx[label]
                if self.transform is not None:
                    image = self.transform(image)
                return image, label


        train_set = MDataset(train_image_paths,
                            transform=transform_train)
        test_set = MDataset(test_image_paths,
                            transform=transform_test)
        self.train_loader = DataLoader(train_set,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers)

        self.test_loader = DataLoader(test_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers)

        #Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        #Setup device
        if args.gpus is not None:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            self.device = 'cuda:' + str(args.gpus[0])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        #Resume checkpoint, if any
        if args.resume:
            print('Loading model checkpoint at: {}'.format(args.resume))
            package = torch.load(args.resume, map_location='cpu')
            model_state_dict = package['state_dict']
            model.load_state_dict(model_state_dict, strict=args.strict)

        if args.gpus is not None and len(args.gpus) == 1:
            model = model.to(device=self.device)
        if args.gpus is not None and len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
        self.model = model

        #Loss function
        if args.loss == 'SqrHinge':
            self.criterion = SqrHingeLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device=self.device)

        #Init optimizer
        if args.optim == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.args.lr,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)

        #Resume optimizer, if any
        if args.resume and not args.evaluate:
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']
            if 'best_val_acc' in package.keys():
                self.best_val_acc = package['best_val_acc']

        #LR scheduler
        if args.scheduler == 'STEP':					#Smart evoluting Learning Rate
            milestones = [int(i) for i in args.milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                         milestones=milestones,
                                         gamma=0.1)
        elif args.scheduler == 'FIXED':		#Not smart evoluting LR/not evoluting LR
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(self.args.scheduler))

        #Resume scheduler, if any
        if args.resume and not args.evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package['epoch'] - 1

    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
        }, best_path)

    def train_model(self):
        
        #Training starts
        if self.args.detect_nan:
            torch.autograd.set_detect_anomaly(True)
        loss_path = os.path.join(self.output_dir_path, "loss_train.txt")

        for epoch in range(self.starting_epoch, self.args.epochs):

            #Set to training mode
            self.model.train()
            self.criterion.train()
            loss_train=[]

            #Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                #For hingeloss only
                if isinstance(self.criterion, SqrHingeLoss):
                    target = target.unsqueeze(1)
                    target_onehot = torch.Tensor(target.size(0), self.num_classes).to(self.device,
                                                                                      non_blocking=True)
                    target_onehot.fill_(-1)
                    target_onehot.scatter_(1, target, 1)
                    target = target.squeeze()
                    target_var = target_onehot
                else:
                    target_var = target

                #Measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                #Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target_var)
                
                #Compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.model.clip_weights(-1, 1)
                #Measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i,
                                                       len(self.train_loader))

                #Training batch ends
                start_data_loading = time.time()
                loss_train.append(loss)
                

            #Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            else:
                # Set the learning rate
                if epoch % self.args.stepLR == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5

    	    #Put the loss in file
            average_loss=Average(loss_train)
            with open(loss_path, 'a') as f:
             f.write(str(average_loss))
             f.write("\n")
             f.close
             
            #Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            #Checkpoint
            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
            elif not self.args.dry_run:
                self.checkpoint_best(epoch, "checkpoint.tar")

        #Training ends
        if not self.args.dry_run:
            return os.path.join(self.checkpoints_dir_path, "best.tar")


    #Function called during the testing session
    def eval_model(self, epoch=None):
        dataset=self.args.dataset
    	#Compute future matrix for results
        if dataset=='MNIST_OddEven':
            A=[[0,0],
               [0,0]]
        else:
            A = [[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]]
        B = []

        eval_meters = EvalEpochMeters()

        #Switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            

            #For hingeloss only
            if isinstance(self.criterion, SqrHingeLoss):
                target = target.unsqueeze(1)
                target_onehot = torch.Tensor(target.size(0), self.num_classes).to(self.device,
                                                                                  non_blocking=True)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target, 1)
                target = target.squeeze()
                target_var = target_onehot
            else:
                target_var = target
           
            #Compute output
            output = self.model(input)
            
            
            if self.args.evaluate:
             for x in range(100): 
               part = self.test[x + (i*100)] + " answer " + str(output[x].numpy()) + "     " + str(np.argmax(output[x].numpy()))
               B.append(part)
              

            #Measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            #Compute loss
            loss = self.criterion(output, target_var)
            eval_meters.loss_time.update(time.time() - end)

            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            #Write the output vector guess of the network
            if self.args.evaluate:
             for x in range(100) : 
               A[target.data.view_as(pred).numpy()[x][0]][pred.numpy()[x][0]] = A[target.data.view_as(pred).numpy()[x][0]][pred.numpy()[x][0]] + 1

            prec1 = 100. * correct.float() / input.size(0)

            _, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))


            #Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))
        if self.args.evaluate:

         test_path = os.path.join(self.output_dir_path, "test.txt")
         matrix_path = os.path.join(self.output_dir_path, "matrix_noisy.txt")
         table_path = os.path.join(self.output_dir_path, "table_noisy.txt")
         loss_path = os.path.join(self.output_dir_path, "loss_test.txt")
         print(test_path)
         with open(test_path, 'a') as f:
             f.write(str(self.model))
             f.close

         #Write results into txt files
         for name, param in self.model.named_parameters():
           with open(test_path, 'a') as f:	#File for weights
             f.write(str(name))
             f.write(":\n")
             f.write(str(param))
             f.write("\n")
             f.close
         with open(matrix_path, 'w') as f:	#File for the matrix
           for line in A:
             f.write(str(line))
             f.write('\n')
             f.close
         with open(loss_path, 'a') as f:	#File for testing loss
             f.write(str(loss))
             f.write("\n")
             f.close
         with open(table_path, 'w') as f:	#File for the output vectors
           for line in B:
             f.write(line)
             f.write('\n')
             f.close
        
        return eval_meters.top1.avg
