import open3d as o3d
import glob
import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as TG
import torchvision.transforms as TV
from torchvision.utils import make_grid, save_image
import torch_geometric.nn as gnn
from torch_geometric.read import read_off
from PIL import Image


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_geometry(xyz, path_to_save):
    xyz = xyz.cpu().detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path_to_save, pcd)


def save_grid(img, grid, path_to_save, nrow=6):
    grid = grid.permute(0, 2, 3, 1).cpu().detach().numpy()
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    t = TV.ToTensor()
    attn_list = []
    for i in range(grid.shape[0]):
        g = grid[i]
        rect = np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]])
        rect = img.shape[:2] * (rect + 1) / 2
        rect = rect.astype(np.int32)
        img_vis = img.copy()
        img_vis = cv2.polylines(img_vis, [rect], isClosed=True, color=(160, 20, 20), thickness=2)
        attn_list.append(t(img_vis))
    save_image(attn_list, path_to_save, nrow=nrow)


def custom_draw_geometry(xyz, path_to_save=None):
    xyz = xyz.cpu().detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    if path_to_save is not None:
        vis.capture_screen_image(filename=path_to_save, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def batch_voxelized_iou(pred_batch, target_batch, voxel_size):
    values = np.array([voxelizedIoU(pred_batch[i], 
                                    target_batch[i], 
                                    voxel_size) for i in range(pred_batch.size(0))])
    return values.mean()


def voxelizedIoU(pred_pcd, target_pcd, voxel_size):
    xyz1 = pred_pcd.cpu().detach().numpy()
    xyz2 = target_pcd.cpu().detach().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    pcd2.points = o3d.utility.Vector3dVector(xyz2)

    # align both pcds to have origin at [-1., -1., -1.]
    transform1 = np.eye(4)
    transform1[:-1, -1] = np.array([-1., -1., -1.]) - pcd1.get_min_bound()
    pcd1.transform(transform1)
    transform2 = np.eye(4)
    transform2[:-1, -1] = np.array([-1., -1., -1.]) - pcd2.get_min_bound()
    pcd2.transform(transform2)

    # create voxels of surfaces
    vxl1 = o3d.geometry.create_surface_voxel_grid_from_point_cloud(pcd1, 
                                                                   voxel_size)
    vxl2 = o3d.geometry.create_surface_voxel_grid_from_point_cloud(pcd2, 
                                                                   voxel_size)

    occupancy_indxs1 = np.array([x.grid_index for x in vxl1.voxels])
    occupancy_indxs2 = np.array([x.grid_index for x in vxl2.voxels])

    max_x, max_y, max_z = np.concatenate([occupancy_indxs1, occupancy_indxs2], axis=0).max(axis=0)

    vxl1 = np.zeros((max_x+1, max_y+1, max_z+1), dtype=bool)
    vxl2 = np.zeros((max_x+1, max_y+1, max_z+1), dtype=bool)
    vxl1 &= False
    vxl2 &= False
    vxl1[occupancy_indxs1[:, 0], 
         occupancy_indxs1[:, 1], 
         occupancy_indxs1[:, 2]] = True
    vxl2[occupancy_indxs2[:, 0], 
         occupancy_indxs2[:, 1], 
         occupancy_indxs2[:, 2]] = True

    # compute iou
    iou = np.sum(vxl1&vxl2) / np.sum(vxl1|vxl2)

    return iou


def cd(pred, targets):
    '''pred: NxMx3, targets: NxKx3''' 
    # (y_ - y)**2 = <y_, y_> - 2 * <y, y_> + <y, y>
    dists = (pred**2).sum(-1, keepdim=True) - \
            2 * pred @ targets.transpose(2, 1) + \
            (targets**2).sum(-1).unsqueeze(-2)
    # dists = torch.sqrt(dists + 1e-6)
    return dists.min(-2)[0].mean() + dists.min(-1)[0].mean()


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


def losses(pred, target, target_norms, device, chamfer=True, edge=False, norm=False, t=2/32, metrics=True):    
    # form vectors of batch indicators
    pred_batch = torch.ones(pred.size()[:2], 
                            dtype=torch.int64,
                            device=device)
    pred_batch *= torch.arange(start=0, 
                               end=pred.size(0), 
                               dtype=torch.int64, 
                               device=device).view(-1, 1)
    pred_batch = pred_batch.flatten()

    target_batch = torch.ones(target.size()[:2], 
                              dtype=torch.int64, 
                              device=device)
    target_batch *= torch.arange(start=0, 
                                 end=target.size(0), 
                                 dtype=torch.int64, 
                                 device=device).view(-1, 1)
    target_batch = target_batch.flatten()

    chamfer_loss = edge_loss = norm_loss = 0.
    precision = recall = fscore = 0.

    # Chamfer Loss
    if chamfer or norm:
        target_pred_nearest = gnn.knn(x=pred.view(-1, 3), 
                                      y=target.view(-1, 3), 
                                      k=1, 
                                      batch_x=pred_batch, 
                                      batch_y=target_batch)
        target_pred_nearest = target_pred_nearest.view(2, target.size(0), target.size(1), -1)

        pred_target_nearest = gnn.knn(x=target.view(-1, 3), 
                                      y=pred.view(-1, 3), 
                                      k=1, 
                                      batch_x=target_batch, 
                                      batch_y=pred_batch)
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
        pred_k_3 = gnn.knn_graph(x=pred.view(-1, 3), 
                                 k=3, 
                                 batch=pred_batch)
        # Edge Loss    
        if edge:
            pred_edge_dist = torch.norm(pred.view(-1, 3)[pred_k_3[0, 0::3]] -
                                        pred.view(-1, 3)[pred_k_3[1, 0::3]], dim=-1)
            pred_edge_dist = pred_edge_dist.view(pred.size(0), pred.size(1))
            edge_loss = pred_edge_dist.mean(dim=-1).mean()
        
        # Norm Loss    
        if norm:
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

    if metrics:
        # compute metrics
        precision = (target_pred_dist <= t).type(torch.float).mean(dim=-1).mean()
        recall = (pred_target_dist <= t).type(torch.float).mean(dim=-1).mean()
        fscore = ((2 * precision * recall) / (precision + recall)).mean()

        # fix nan. values
        precision[precision != precision] = 0.
        recall[recall != recall] = 0.
        fscore[fscore != fscore] = 0.

    losses_all = {'cd': chamfer_loss,
                  'el': edge_loss,
                  'nl': norm_loss}
    metrics_all = {'precision': precision,
                   'recall': recall,
                   'fscore': fscore}

    return losses_all, metrics_all


def sample_minibatch_from_sphere(k, num_points):
    # torch.random.manual_seed(torch.random.default_generator.seed() // 17)
    # torch.random.manual_seed(42) 

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
                 name='10',
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
        if pts_to_save is not None and pts_to_save > 0:
            self.pcd_transforms = [self.get_view_transform(i, pts_to_save) for i in range(12)]
        if cache_pcds:
             super(Im2PCD, self).__init__(pcd_root, name, train, self.pcd_transforms[0], 
                                          pcd_pre_transform, pcd_pre_filter)
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
        # return 1
        return sum(self.categories_cap) * 12
        # return self.categories_cap[self.categories.index('table')] * 12
    
    def __getitem__(self, idx):
        # torch.random.manual_seed(42)
        # idx = idx + sum(self.categories_cap[:self.categories.index('table')]) * 12
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
            'Only integers, slices (`:`) and long or byte tensors are valid indices (got {}).'.format(type(idx).__name__))
