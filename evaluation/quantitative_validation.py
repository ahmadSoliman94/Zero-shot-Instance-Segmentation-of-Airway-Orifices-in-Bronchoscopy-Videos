import warnings

import torch
from matplotlib import pyplot as plt
from skimage.measure import find_contours
from torch.cuda import amp
from torch.nn import functional as F
from tqdm import tqdm

import sys
sys.path.append('./architectures/')
from architectures.lr_aspp import LiteRASPP


sys.path.append('./datasets/')
from datasets import phantom_dataset
#from datasets.real.real_dataset import RealSegDataset
from datasets.cvc_dataset import CVC_Dataset

import utils
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model = LiteRASPP.load(f'./segment_model/fine_tuning.pt', device)
model.eval()

# Manually set hyperparameters
hp = {
  'img_resolution': 256,  # Example default value, adjust as needed
  'gray_scaled': False    # Example default value, adjust as needed
}

spatial_dim = hp['img_resolution']
gray_scaled = hp['gray_scaled']

#ds = phantom_dataset.PhantomSegDataset('blob', 'val', spatial_dim, normalize=True, gray_scaled=gray_scaled)
#ds = CVC_Dataset(spatial_dim, normalize=True, gray_scale=gray_scaled)
ds = CVC_Dataset(spatial_dim, normalize=True, gray_scale=gray_scaled)

in_mm = False

if isinstance(ds, phantom_dataset.PhantomSegDataset) and in_mm:
    z = phantom_dataset.PhantomSegDataset('z', 'test', spatial_dim).gt
    if len(ds) != len(z):
        raise ValueError('length not equal. Please check if the modes match.')

num_diff = torch.empty(len(ds), dtype=torch.int8)
dice = torch.empty_like(num_diff, dtype=torch.float)
c_dist = torch.empty_like(dice)
for i, (x, y) in enumerate(tqdm(ds)):
    with amp.autocast(), torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        y_hat = F.interpolate(logits, size=y.shape[-2:], mode='bilinear').argmax(1).squeeze(0)

    y[y == -1] = 0
    y = y.squeeze(0)
    y_hat = y_hat.squeeze(0).cpu()

    assert torch.all(torch.isin(y.to(int), torch.tensor([0, 1])))

    # extract airways
    label_y, c_y, num_y = utils.extract_airways(y.to(torch.bool).numpy())
    label_y_hat, c_y_hat, num_y_hat = utils.extract_airways(y_hat.to(torch.bool).numpy())

    # calculate dice for binary label
    if num_y > 0:
        dice[i] = utils.dice_coeff(y_hat, y, max_label=1)
    else:
        dice[i] = - 1

    # calculate centroids distances
    if num_y > 0 and num_y_hat > 0:
        if isinstance(ds, phantom_dataset.PhantomSegDataset) and in_mm:
            # extract Z for rescaling
            z_y = torch.empty(num_y)
            contours = find_contours(y.numpy(), fully_connected='high')

            # plot contours
            # if len(contours) != num_y:
            #     plt.imshow(label_y)
            #     color_list = ['r', 'g', 'b']
            #     for c_i, c in enumerate(contours):
            #         print(c_i)
            #         plt.scatter(c[:, 1], c[:, 0], c=color_list[c_i])
            #     plt.show()

            assert len(contours) == num_y
            for c_idx in range(len(contours)):
                z_contour = utils.sample_from_image(z[i], torch.from_numpy(contours[c_idx]).flip(-1))
                z_y[c_idx] = z_contour.median()

                # convert to world coordinates x_hat
                c_y = phantom_dataset.pixel_to_normalized_world_coordinates(torch.as_tensor(c_y),
                                                                            [spatial_dim, spatial_dim])
                c_y_hat = phantom_dataset.pixel_to_normalized_world_coordinates(torch.as_tensor(c_y_hat),
                                                                                [spatial_dim, spatial_dim])

                # calculate centroids distances
                d = c_y.unsqueeze(1) - c_y_hat.unsqueeze(0)  # nxmxd
                d = d.norm(p=2, dim=-1).min(1)[0]

                c_dist[i] = (d * z_y * phantom_dataset.Z_MAX_DEPTH).mean()
        else:
            d = torch.from_numpy(c_y).unsqueeze(1) - torch.from_numpy(c_y_hat).unsqueeze(0)  # nxmxd
            d = d.norm(p=2, dim=-1).min(1)[0]

            c_dist[i] = d.mean()
    else:
        c_dist[i] = -1

    num_diff[i] = num_y_hat - num_y

# remove samples without proper ground truth
dice = dice[dice != -1]
c_dist = c_dist[c_dist != -1]


print('data set', ds)
print(f'num diff centroids: {num_diff.float().mean().item()} +- {num_diff.float().std().item()}')
print(f'DSC: {dice.mean().item()} +- {dice.std().item()}, median: {dice.median().item()}')
if in_mm and isinstance(ds, phantom_dataset.PhantomSegDataset):
    print(f'D_c[mm]: {c_dist.mean().item()} +- {c_dist.std().item()}, median: {c_dist.median().item()}')
else:
    print(f'D_c[px]: {c_dist.mean().item()} +- {c_dist.std().item()}, median: {c_dist.median().item()}')

plt.hist(num_diff.numpy(), bins='auto')
plt.xlabel('|C_predicted|-|C_ground truth|')
plt.figure()
b = plt.boxplot(dice.numpy(), showfliers=False, notch=True, showmeans=True)
plt.title('Dice mean')
plt.figure()
plt.boxplot(c_dist.numpy(), showfliers=False, notch=True, showmeans=True)
plt.title('Mean centroid distance')
plt.ylabel('distance [mm]')

plt.show()
