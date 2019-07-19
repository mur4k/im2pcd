from data_utils import *
from im_2_pcd_conv import Im2PcdConv
import torch
import torchvision.transforms as TV
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

img_transform = TV.Compose([TV.ToTensor()])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('model_conv.obj')
model.to(device)
model.eval()


test_set = Im2PCD('../../data/modelnet10_images_new_12x/',
                  '../../data/ModelNet10',
                  train=False,
                  cache_pcds=True,
                  generate_norms=True,
                  img_transform=img_transform,
                  pts_to_save=14*14*6)

test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# build embeddings visualization
writer = SummaryWriter()

images = []
embed_vectors = []
cnt = 0
for i, data in enumerate(test_loader):
    inputs, pcd, pcd_norms = data
    inputs = inputs.to(device)
    pcd = pcd.to(device)
    pcd_norms = pcd_norms.to(device)
    embeddings = model.encoder_block1(inputs)
    embeddings = model.encoder_block2(embeddings)
    embeddings = model.encoder_block3(embeddings)
    embeddings = model.encoder_block4(embeddings)
    embeddings = model.encoder_block5(embeddings)
    embeddings = model.avgpool(embeddings).view(embeddings.size(0), -1)
    embeddings = model.fcn_encoder(embeddings)
    images.append(inputs.to(torch.device('cpu')))
    embed_vectors.append(embeddings.to(torch.device('cpu')))
    cnt += inputs.size(0)
    if cnt >= 100:
        break
images = torch.cat(images, dim=0)
embed_vectors = torch.cat(embed_vectors, dim=0)
print('create embeding visualization of {} samples...'.format(cnt))
writer.add_embedding(embed_vectors, label_img=images)
print('finished')

# build pcd & attention visualizations
print('save and visualize input, output and attention activations...')
inputs, pcd, pcd_norms = iter(test_loader).next()
inputs = inputs.to(device)
pcd = pcd.to(device)
pcd_norms = pcd_norms.to(device)

# encoder
x_enc1 = model.encoder_block1(inputs)
x_enc2 = model.encoder_block2(x_enc1)
x_enc3 = model.encoder_block3(x_enc2)
x_enc4 = model.encoder_block4(x_enc3)
x_enc5 = model.encoder_block5(x_enc4)
x_enc5 = model.avgpool(x_enc5)
x_enc5 = x_enc5.view(x_enc5.size(0), -1)
x_enc5 = model.fcn_encoder(x_enc5)
x_enc5 = x_enc5.unsqueeze(-1).unsqueeze(-1)

# decoder
outputs = model.decoder_block5(x_enc5)
outputs = model.decoder_block4(outputs)
outputs, grid1 = model.spfp_block3(outputs, x_enc3)
outputs, grid2 = model.spfp_block2(outputs, x_enc2)
outputs, grid3 = model.spfp_block1(outputs, x_enc1)
outputs = model.fcn_decoder(outputs)
yv, xv = torch.meshgrid([torch.linspace(-1, 1, 14), torch.linspace(-1, 1, 14)])
yv = yv.to(model.device)
xv = xv.to(model.device)
outputs[:, 0::3, :, :] += xv
outputs[:, 1::3, :, :] += yv
outputs = outputs.permute(0, 2, 3, 1)
outputs = outputs.contiguous().view(outputs.size(0), -1, 3)

ensure_dir('./val_vis/input_image_000000001.png')
for i in range(inputs.size(0)):
    grid = torch.cat([grid1[i], grid2[i], grid3[i]], dim=0)
    save_image(inputs[i], 'val_vis/{:09d}_inp_img.png'.format(i))
    save_geometry(pcd[i], 'val_vis/{:09d}_inp_pcd.pcd'.format(i))
    save_geometry(outputs[i], 'val_vis/{:09d}_outp_pcd.pcd'.format(i))
    save_grid(inputs[i], grid, 'val_vis/attnt_grid_{:09d}.png'.format(i), nrow=6)
print('finished')
