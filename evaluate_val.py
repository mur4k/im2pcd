from data_utils import *
from im_2_pcd_conv import Im2PcdConv
import torch
import torchvision.transforms as TV
from torch.utils.data import DataLoader
import pandas as pd


def loss(pred, target, target_norms, device):
    l, m = losses(pred, target, target_norms, device)
    return 1. * l['cd'] + 0.5 * l['el'] + 0.25 * l['nl'], m['fscore']


img_transform = TV.Compose([TV.ToTensor()])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('model_conv.obj')
model.to(device)
model.eval()

test_set = Im2PCD('../../data/modelnet10_images_new_12x/',
                  '../../data/ModelNet10',
                  train=False,
                  cache_pcds=False,
                  generate_norms=True,
                  img_transform=img_transform,
                  pts_to_save=14*14*6)

metrics = {'Loss': [], 'F_score': [], 'IoU': []}
classes = test_set.categories

for i, c in enumerate(classes):
	print('computing metrics for class {}'.format(c))
	start_idx = sum(test_im2pcd.categories_cap[:i]) * 12
	end_idx = start_idx + test_im2pcd.categories_cap[i] * 12
	data_loader = DataLoader(test_im2pcd[start_idx:end_idx], batch_size=16)

	running_loss = 0.0
    running_iou = 0.0
    running_fscore = 0.0
  
    # Iterate over data.
    for i, data in enumerate(data_loader):
        
        # get the inputs
        inputs, pcd, pcd_norms = data
        inputs = inputs.to(device)
        pcd = pcd.to(device)
        pcd_norms = pcd_norms.to(device)

        # Get model outputs and calculate loss
        outputs = model(inputs)
        loss, fscore = loss(outputs, pcd, pcd_norms, device)

        # compute iou
        iou = batch_voxelized_iou(outputs, pcd, voxel_size=2/32)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_iou += iou * inputs.size(0)
        running_fscore += fscore * inputs.size(0)

    class_loss = running_loss / len(data_loader.dataset)
    class_iou = running_iou / len(data_loader.dataset)
    class_fscore = running_fscore / len(data_loader.dataset)

    metrics['Loss'].append(class_loss)
    metrics['F_score'].append(class_fscore)
    metrics['IoU'].append(class_iou)

report = pd.DataFrame(data=metrics, index=classes)
report.to_csv('classwise_metrics.csv')