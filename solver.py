from random import shuffle
import numpy as np
import time
import copy
import gc
import sys
import argparse
import shutil

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from torch.utils.data import DataLoader
import torchvision.transforms as TV

from logger import Logger
from data_utils import *
from im_2_pcd_conv import Im2PcdConv
from im_2_pcd_graph import Im2PcdGraph



class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self._reset_histories()
        self.writer = SummaryWriter()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, model, dataloaders, start_epoch, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000.0
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model.to(device)

        iter_per_epoch = {mode: len(dataloaders[mode]) for mode in ['train', 'val']}

        print('START TRAIN.')

        # loop over the dataset multiple times
        for epoch in range(num_epochs):  

            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
      
                torch.random.manual_seed(torch.random.default_generator.seed() // 17) 

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    
                    # get the inputs
                    inputs, pcd, pcd_norms = data
                    inputs = inputs.to(device)
                    pcd = pcd.to(device)
                    pcd_norms = pcd_norms.to(device)

                    # zero the parameter gradients
                    optim.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = self.loss_func(outputs, pcd, pcd_norms)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optim.step()
                            self.train_loss_history.append(loss.item())

                    if (i + 1) % log_nth == 0:    # print every log_nth mini-batches
                        step = (start_epoch + epoch) * iter_per_epoch[phase] + i

                        print('[Iteration {}/{}] {} loss: {:.4f}'.format(i + 1, iter_per_epoch[phase], phase, loss))

                        # ================================================================== #
                        #                        Tensorboard Logging                         #
                        # ================================================================== #

                        # 1. Log scalar values (scalar summary)
                        info = {'{} loss'.format(phase): loss.item()}

                        for tag, value in info.items():
                            self.writer.add_scalar(tag, value, step + 1)

                        if phase == 'train':

                            # 2. Log values and gradients of the parameters (histogram summary)
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                self.writer.add_histogram(tag, value.data.cpu().numpy(), step + 1)
                                self.writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

                        # # 3. Log training images (image summary)
                        # info = {'images': inputs[:10].cpu().numpy()}

                        # for tag, images in info.items():
                        #     for i in range(images.shape[0]):
                        #         self.writer.add_image(tag, images[i], step + 1)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    self.val_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)


if __name__ == 'main':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to images")
    ap.add_argument("-p", "--points", required=True,
        help="path to pcds")
    args = vars(ap.parse_args())

    # img_transform = TV.Compose([TV.ToTensor(), TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform = TV.Compose([TV.ToTensor()])

    train_im2pcd = Im2PCD(args['images'],
                          args['points'],
                          train=True,
                          cache_pcds=True,
                          generate_norms=True,
                          img_transform=img_transform,
                          pts_to_save=14*14*6)
    test_im2pcd = Im2PCD(args['images'],
                         args['points']
                         train=False,
                         cache_pcds=True,
                         generate_norms=True,
                         img_transform=img_transform,
                         pts_to_save=14*14*6)

    train_loader = DataLoader(train_im2pcd, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_im2pcd, batch_size=1, shuffle=True)
    dataloaders = {'train': train_loader,
                   'val': test_loader}

    model = Im2PcdGraph(num_points=14*14*6)
    k = 0
    for p in model.parameters():
        k += p.size().numel()
    print(model)
    print('Number of parameters: {}'.format(k))

    shutil.rmtree('./runs', ignore_errors=True)

    def loss(pred, target, target_norms):
        cl, el, nl = losses(pred, target, target_norms, 1, 0, 0)
        return 1. * cl + 0. * el + 0. * nl

    solver = Solver(optim_args={"lr": 3e-4, "weight_decay": 0.0}, loss_func=loss)

    solver.train(model, dataloaders, log_nth=1, start_epoch=0, num_epochs=100)