from pathlib import Path

import torch
from matplotlib import pyplot as plt
from skimage.measure import find_contours
from torch.cuda import amp
from torch.nn import functional as F
from tqdm import tqdm

from architectures.lr_aspp import LiteRASPP
from datasets.phantom import phantom_dataset
from segmentation import utils

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model_name = 'bseg_weight_decay'
model = LiteRASPP.load(f'./segmentation/results/{model_name}/fine_tuning.pt', device)
model.eval()
spatial_dim = model.spatial_dim
c_mask = utils.create_circular_mask(spatial_dim, spatial_dim, exclude_border=True)

# load data
img = []
seg_mask = []
depth_img = []

base = Path(f'./datasets/phantom/data/tensor_data/{spatial_dim}x{spatial_dim}')
for n in tqdm(phantom_dataset.dataset_sequence_splits(mode='test'), desc='load data'):
    img.append(torch.load(base.joinpath(f'{n}_aligned_rgb.pt')))
    seg_mask.append(torch.load(base.joinpath(f'{n}_blob.pt')))
    depth_img.append(torch.load(base.joinpath(f'{n}_z.pt')))
img = torch.cat(img, dim=0).float().div(255)
seg_mask = torch.cat(seg_mask, dim=0)
depth_img = torch.cat(depth_img, dim=0)
assert len(img) == len(seg_mask) == len(depth_img)

# normalize
img = (img - phantom_dataset.MU.view(1, 3, 1, 1)) / phantom_dataset.SIGMA.view(1, 3, 1, 1)

num_diff = torch.empty(len(img), dtype=torch.int8)
dice = torch.empty_like(num_diff, dtype=torch.float)
c_dist = torch.empty_like(dice)
for i, (x, y, z) in enumerate(
        tqdm(zip(img, seg_mask, depth_img), total=len(img))):  # x: input, y: ground truth, z: depth_image
    with amp.autocast(), torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        y_hat = F.interpolate(logits, size=y.shape[-2:], mode='bilinear').argmax(1).squeeze(0)

    # exclude circle background
    y = y.squeeze(0)
    y[c_mask] = 0
    y_hat = y_hat.squeeze(0).cpu()
    y_hat[c_mask] = 0

    # extract airways
    label_y, c_y, num_y = utils.extract_airways(y.to(torch.bool).numpy())
    label_y_hat, c_y_hat, num_y_hat = utils.extract_airways(y_hat.to(torch.bool).numpy())

    # calculate dice for binary label
    dice[i] = utils.dice_coeff(y_hat, y, max_label=1)

    # extract Z for rescaling
    z_y = torch.empty(num_y)
    contours = find_contours(y.numpy(), fully_connected='high')

    # from matplotlib import pyplot as plt
    # plt.imshow(label_y)
    # color_list = ['r', 'g', 'b']
    # for c_i, c in enumerate(contours):
    #     print(c_i)
    #     plt.scatter(c[:, 1], c[:, 0], c=color_list[c_i])
    # plt.show()

    assert len(contours) == num_y
    for c_idx in range(len(contours)):
        z_contour = utils.sample_from_image(z, torch.from_numpy(contours[c_idx]).flip(-1))
        z_y[c_idx] = z_contour.median()

    # convert to world coordinates x_hat
    c_y = phantom_dataset.pixel_to_normalized_world_coordinates(torch.from_numpy(c_y), [spatial_dim, spatial_dim])
    c_y_hat = phantom_dataset.pixel_to_normalized_world_coordinates(torch.from_numpy(c_y_hat),
                                                                    [spatial_dim, spatial_dim])

    # calculate centroids distances
    d = c_y.unsqueeze(1) - c_y_hat.unsqueeze(0)  # nxmxd
    d = d.norm(p=2, dim=-1).min(1)[0]

    c_dist[i] = (d * z_y * phantom_dataset.Z_MAX_DEPTH).mean()

    num_diff[i] = num_y_hat - num_y

print(f'DCE: {dice.mean().item()}+-{dice.std().item()}')
print(f'D_c: {c_dist.mean().item()}+-{c_dist.std().item()}')

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
