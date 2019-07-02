from data_utils import *
from logger import Logger
from im_2_pcd_conv import Im2PcdConv
from im_2_pcd_graph import Im2PcdGraph

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
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

    def train(self, model, dataloaders, start_epoch, num_epochs=10, log_nth=0, img_to_track_progress=None):
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model.to(device)

        if img_to_track_progress is not None:
            ensure_dir('./model_progress')
            img_to_track_progress = img_to_track_progress.to(device).unsqueeze(0)

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
                        loss = self.loss_func(outputs, pcd, pcd_norms, device)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optim.step()
                            
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
                if phase == 'val':
                    self.val_loss_history.append(epoch_loss)
                    # store best model
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                    # track progress
                    if img_to_track_progress is not None:
                        pcd_pred = model(img_to_track_progress).squeeze(0)
                        path_to_save_progress = './model_progress/{0:09}.pcd'.format(epoch+1) 
                        save_geometry(pcd_pred, path_to_save=path_to_save_progress)
                else:
                    self.train_loss_history.append(epoch_loss)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)


def loss(pred, target, target_norms, device):
    cd, el, nl = losses(pred, target, target_norms, device, 1, 0, 0)
    return 1. * cd + 0.5 * el + 0.25 * nl

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-x", "--images", required=True,
        help="path to images")
    ap.add_argument("-y", "--points", required=True,
        help="path to pcds")
    ap.add_argument("-m", "--path_to_save_model", required=True,
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
                         args['points'],
                         train=False,
                         cache_pcds=True,
                         generate_norms=True,
                         img_transform=img_transform,
                         pts_to_save=14*14*6)

    train_loader = DataLoader(train_im2pcd, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_im2pcd, batch_size=16, shuffle=True)
    dataloaders = {'train': train_loader,
                   'val': test_loader}

    model = Im2PcdConv(num_points=14*14*6)
    k = 0
    for p in model.parameters():
        k += p.size().numel()
    print(model)
    print('Number of parameters: {}'.format(k))

    shutil.rmtree('./runs', ignore_errors=True)

    solver = Solver(optim_args={"lr": 1e-4, "weight_decay": 1e-5}, loss_func=loss)

    img_progress, pcd_progress, pcd_norms_progress = train_im2pcd[0]
    solver.train(model, 
                 dataloaders, 
                 log_nth=1, 
                 start_epoch=0, 
                 num_epochs=100, 
                 img_to_track_progress=img_progress)

    model.save(args['path_to_save_model'])

    plt.plot(solver.train_loss_history)
    plt.plot(solver.val_loss_history)
    plt.legend(['train loss', 'validation loss'])
    plt.title('Im2PCD')
    plt.savefig('learning_curve.png')
