import glob
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as TG
import torchvision.transforms as TV
import torch_geometric.nn as gnn
from torch_geometric.read import read_off
from PIL import Image


def onedir_nn_distance(p1, p2, norm=True):
    p1_copy = p1.repeat(p2.size(0), 1, 1).transpose(0, 1)
    p1_res = (p1_copy - p2).norm(dim=-1).min(dim=-1).values
    return p1_res.mean() if norm else p1_res.sum()

def chamfer_distance(p1, p2, norm=True):
    return onedir_nn_distance(p1, p2, norm) + onedir_nn_distance(p2, p1, norm)

def edge_loss(p):
    """
    p: BxNx3
    """
    batch = torch.cat([torch.ones(p.size(1)) * bn for bn in range(p.size(0))])
    nearest_idxs = gnn.knn(p.view(-1, 3), p.view(-1, 3), 2, batch, batch)[1, 1::2]
    edge_len = torch.norm(p.view(-1, 3) - p.view(-1, 3)[nearest_idxs], dim=-1).view(p.size()[:2])
    edge_loss = edge_len.mean(dim=-1).mean()
    return edge_loss

def losses(pred, target, target_norms, chamfer=True, edge=False, norm=False):

    # form vectors of batch indicators
    pred_batch = torch.ones(pred.size()[:2])
    pred_batch *= torch.arange(start=0, end=pred.size(0), dtype=torch.float32).view(-1, 1)

    target_batch = torch.ones(target.size()[:2])
    target_batch *= torch.arange(start=0, end=target.size(0), dtype=torch.float32).view(-1, 1)

    chamfer_loss = edge_loss = norm_loss = 0.

    # Chamfer Loss

    if chamfer or norm:

        target_pred_nearest = gnn.knn(x=pred.view(-1, 3), 
                                      y=target.view(-1, 3), 
                                      k=1, 
                                      batch_x=pred_batch.flatten(), 
                                      batch_y=target_batch.flatten())
        target_pred_nearest = target_pred_nearest.view(2, target.size(0), target.size(1), -1)

        pred_target_nearest = gnn.knn(x=target.view(-1, 3), 
                                      y=pred.view(-1, 3), 
                                      k=1, 
                                      batch_x=target_batch.flatten(), 
                                      batch_y=pred_batch.flatten())
        pred_target_nearest = pred_target_nearest.view(2, pred.size(0), pred.size(1), -1)

        if chamfer:

            pred_target_dist = torch.norm(pred.view(-1, 3)[pred_target_nearest[0].view(-1)] - 
                                          target.view(-1, 3)[pred_target_nearest[1].view(-1)], dim=-1)
            pred_target_dist = pred_target_dist.view(pred.size(0), pred.size(1))

            target_pred_dist = torch.norm(target.view(-1, 3)[target_pred_nearest[0].view(-1)] - 
                                          pred.view(-1, 3)[target_pred_nearest[1].view(-1)], dim=-1)
            target_pred_dist = target_pred_dist.view(target.size(0), target.size(1))

            target_pred_dist_mean = target_pred_dist.mean(dim=-1).mean()
            pred_target_dist_mean = pred_target_dist.mean(dim=-1).mean()

            chamfer_loss = pred_target_dist_mean + target_pred_dist_mean
    
    if edge or norm: 
    
        pred_k_3 = gnn.knn_graph(x=pred.view(-1, 3), k=3, batch=pred_batch.flatten(), loop=False)

        if edge:

            # Edge Loss

            pred_edge_dist = torch.norm(pred.view(-1, 3)[pred_k_3[0, 0::3]] -
                                        pred.view(-1, 3)[pred_k_3[1, 0::3]], dim=-1)
            pred_edge_dist = pred_edge_dist.view(pred.size(0), pred.size(1))
            edge_loss = pred_edge_dist.mean(dim=-1).mean()

        if norm:

            # Norm Loss

            pred_vec1 = pred.view(-1, 3)[pred_k_3[1, 1::3]] - pred.view(-1, 3)[pred_k_3[1, 0::3]]
            pred_vec2 = pred.view(-1, 3)[pred_k_3[1, 2::3]] - pred.view(-1, 3)[pred_k_3[1, 0::3]]

            # compute normal vector
            pred_approx_norm = torch.cross(pred_vec1, pred_vec2)
            pred_approx_norm = pred_approx_norm / torch.norm(pred_approx_norm, dim=-1).view(-1, 1)

            # cosine dissimilarity of norms
            pred_target_norm_dissim = 1 - torch.mul(pred_approx_norm[pred_target_nearest[0].view(-1)], 
                                                    target_norms.view(-1, 3)[pred_target_nearest[1].view(-1)]).sum(dim=-1).abs()
            pred_target_norm_dissim = pred_target_norm_dissim.view(pred.size(0), pred.size(1))

            target_pred_norm_dissim = 1 - torch.mul(target_norms.view(-1, 3)[target_pred_nearest[0].view(-1)], 
                                                    pred_approx_norm[target_pred_nearest[1].view(-1)]).sum(dim=-1).abs()
            target_pred_norm_dissim = target_pred_norm_dissim.view(target.size(0), target.size(1))

            target_pred_norm_dissim_mean = target_pred_norm_dissim.mean(dim=-1).mean()
            pred_target_norm_dissim_mean = pred_target_norm_dissim.mean(dim=-1).mean()

            norm_loss = pred_target_norm_dissim_mean + target_pred_norm_dissim_mean

    return chamfer_loss, edge_loss, norm_loss

def cd(pred, targets):
    '''pred: NxMx3, targets: NxKx3''' 
    # (y_ - y)**2 = <y_, y_> - 2 * <y, y_> + <y, y>
    dists = (pred**2).sum(-1, keepdim=True) - \
            2 * pred @ targets.transpose(2, 1) + \
            (targets**2).sum(-1).unsqueeze(-2)
    # dists = torch.sqrt(dists + 1e-6)
    return dists.min(-2)[0].mean() + dists.min(-1)[0].mean()

def sample_minibatch_from_sphere(k, num_points):
    # torch.random.manual_seed(torch.random.default_generator.seed() // 17)
    torch.random.manual_seed(42) 

    phi = torch.rand(k, 1, num_points) * np.pi
    theta = torch.rand(k, 1, num_points) * 2 * np.pi
    sample = torch.cat([torch.sin(phi) * torch.cos(theta), 
                        torch.sin(phi) * torch.sin(theta), 
                        torch.cos(phi)], dim=1)
    return sample

def x_rotation_matrix(x_axis_deg):
    return torch.tensor([[1, 0, 0], 
                 [0, np.cos(x_axis_deg), -np.sin(x_axis_deg)], 
                 [0, np.sin(x_axis_deg), np.cos(x_axis_deg)]], dtype=torch.float64)


def y_rotation_matrix(y_axis_deg):
    return torch.tensor([[np.cos(y_axis_deg), 0, np.sin(y_axis_deg)], 
                 [0, 1, 0], 
                 [-np.sin(y_axis_deg), 0, np.cos(y_axis_deg)]], dtype=torch.float64)


def z_rotation_matrix(z_axis_deg):
    return torch.tensor([[np.cos(z_axis_deg), -np.sin(z_axis_deg), 0], 
                 [np.sin(z_axis_deg), np.cos(z_axis_deg), 0], 
                 [0, 0, 1]], dtype=torch.float64)


def rotation_matrix(x_axis_deg, y_axis_deg, z_axis_deg):
    return torch.matmul(z_rotation_matrix(z_axis_deg),
                        torch.matmul(y_rotation_matrix(y_axis_deg), 
                                     x_rotation_matrix(x_axis_deg)))

class Im2PCD(ModelNet):
    def __init__(self, 
                 img_root, 
                 pcd_root, 
                 name='40',
                 train=True,
                 pts_to_save=None,
                 img_transform=None, 
                 pcd_pre_transform=None,
                 pcd_pre_filter=None,
                 cache_pcds=False,
                 generate_norms=False):
        self.img_root = img_root
        self.img_transform = img_transform
        self.dataset = 'train' if train else 'test'
        self.root = os.path.expanduser(os.path.normpath(pcd_root))
        self.raw_dir = os.path.join(self.root, 'raw')
        self.pcd_transforms = [None] * 12        
        self.transform = self.pcd_transforms[0]
        self.pre_transform = pcd_pre_transform
        self.pre_filter = pcd_pre_filter
        self.pts_to_save = pts_to_save
        self.generate_norms = generate_norms
        self.cache_pcds = cache_pcds
        self.categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        self.categories = sorted([x.split(os.sep)[-2] for x in self.categories])
        self.categories_cap = []
        self.categories_min = [1] * len(self.categories)
        for i, cat in enumerate(self.categories):
            folder = os.path.join(self.raw_dir, cat, self.dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, cat))
            if self.dataset == 'test':
                self.categories_min[i] = int(os.path.basename(min(paths))[-8:-4])
            self.categories_cap.append(len(paths))
        if pts_to_save is not None and pts_to_save > 0:
            self.pcd_transforms = [self.get_view_transform(i, pts_to_save) for i in range(12)]
        if cache_pcds:
            super(Im2PCD, self).__init__(pcd_root, name, train, self.pcd_transforms[0], 
                                         pcd_pre_transform, pcd_pre_filter)
            
        
    def get_view_transform(self, k, num_pts):
        R = rotation_matrix(np.pi/3., 0., np.pi/6. * k)
        transformation = TG.Compose([TG.NormalizeScale(), 
                                     TG.LinearTransformation(R),
                                     TG.SamplePoints(num=num_pts, include_normals=self.generate_norms)])
        return transformation
    
    def get_image_name(self, cat_num, model_num, view_num):
        return self.categories[cat_num] + \
                '_' + str(model_num).rjust(4, '0') + \
                '.obj.shaded_v' + str(view_num+1).rjust(3, '0') + '.png'
    
    def get_image(self, cat_num, model_num, view_num):
        img_path = os.path.join(self.img_root, 
                                self.categories[cat_num], 
                                self.dataset,
                                self.get_image_name(cat_num, model_num, view_num))
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def get_model_name(self, cat_num, model_num):
        return self.categories[cat_num] + '_' + str(model_num).rjust(4, '0') + '.off' 

    def get_pcd(self, cat_num, model_num):

        path = os.path.join(self.raw_dir, 
                            self.categories[cat_num], 
                            self.dataset,
                            self.get_model_name(cat_num, model_num))
        data = read_off(path)
        data.y = torch.tensor([cat_num, model_num])
        if self.pre_filter is not None:
            data = self.pre_filter(data)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data
       
    def process_set(self, dataset):
        print('xyu')
        if self.cache_pcds:
            data_list = []
            for target, category in enumerate(self.categories):
                folder = os.path.join(self.raw_dir, category, dataset)
                paths = sorted(glob.glob('{}/{}_*.off'.format(folder, category)))
                for path in paths:
                    data = read_off(path)
                    model_num = int(os.path.basename(path)[-8:-4])
                    data.y = torch.tensor([target, model_num])
                    data_list.append(data)                    
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            return self.collate(data_list)
        else:
            pass
    
    def __len__(self):
        return 1
        # return  super(Im2PCD, self).__len__() * 12
    
    def __getitem__(self, idx):

        torch.random.manual_seed(42)

        if isinstance(idx, int):
            model_idx = idx // 12
            view_idx = idx % 12
            if self.cache_pcds:
                data = self.get(model_idx)
                cat_num, model_num = data.y.tolist()
            else:
                cat_num = 0
                acc_cap = 0
                prev_acc_cap = 0
                for i, cap in enumerate(self.categories_cap):
                    prev_acc_cap = acc_cap
                    acc_cap += cap
                    if model_idx < acc_cap:
                        cat_num = i
                        break
                model_num = model_idx - prev_acc_cap + self.categories_min[cat_num]
                data = self.get_pcd(cat_num, model_num)
            data = data if self.pcd_transforms[view_idx] is None else self.pcd_transforms[view_idx](data)
            img = self.get_image(cat_num, model_num, view_idx)
            img = img if self.img_transform is None else self.img_transform(img)                        
            return img, data.pos, data.norm      
        elif isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]        
        raise IndexError(
            'Only integers, slices (`:`) and long or byte tensors are valid '
            'indices (got {}).'.format(type(idx).__name__))