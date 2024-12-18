
import warnings
from os import getcwd
from os.path import isdir, join

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import create_circular_mask
from torchvision.transforms.functional import rgb_to_grayscale

# camera parameter
Z_MAX_DEPTH = 140.91772  # in mm
K = torch.tensor([[236.99869, 0, 210.73509],
                  [0, 237.73802, 191.91745],
                  [0, 0, 1]])  # intrinsic matrix
K_inv = torch.inverse(K)
NATIVE_RESOLUTION = 400  # 400x400

# constants obtained over training split
MU = torch.tensor([0.2780, 0.1812, 0.2275])
SIGMA = torch.tensor([0.2084, 0.1753, 0.2176])

BINARY_CLASS_WEIGHTS = torch.tensor([164119270,  40229413], dtype=torch.float)
BINARY_CLASS_WEIGHTS = BINARY_CLASS_WEIGHTS.div(BINARY_CLASS_WEIGHTS.sum()).pow(-1).sqrt()
# old and deprecated
MULTI_CLASS_WEIGHTS = torch.tensor([269, 8680, 14272, 6665, 1325, 35], dtype=torch.float)
MULTI_CLASS_WEIGHTS = MULTI_CLASS_WEIGHTS.div(MULTI_CLASS_WEIGHTS.sum()).pow(-1).sqrt()

# contained via greedy_search_best_plit.py
def dataset_sequence_splits(mode: str):
    if mode == 'train':
        sample_n = [3, 4, 5, 6, 8, 15]
    elif mode == 'val':
        sample_n = [2, 9, 10, 11, 12, 13, 14, 16]
    elif mode == 'test':
        sample_n = [0, 7]
    else:
        raise ValueError(f'{mode} is an unknown mode.')

    return sample_n


def pixel_to_normalized_world_coordinates(x: torch.Tensor, resolution: (int, int)):
    """
    covert pixel coordinates to normalized world coordinates
    :param x: pixel coordinates [Nx2]
    :param resolution: the resolution of the coordinate's image
    :return: world normalized coordinates [X/Z, Y/Z]
    """
    # subtract from all resolution 1 to obtain index indices
    spatial_dim = torch.as_tensor(resolution).unsqueeze(0) - 1
    # scale back to native resolution
    x = x.float() / spatial_dim * (NATIVE_RESOLUTION - 1)
    # add ones: [x, y, 1]
    x = torch.cat([x, torch.ones(len(x), 1)], dim=-1)
    # x_hat = K^-1 * c with x_hat = [X/Z, Y/Z, 1]
    x_hat = torch.bmm(K_inv.unsqueeze(0).expand(len(x), -1, -1), x.unsqueeze(-1)).squeeze(-1)

    return x_hat[:, :2]


class PhantomSegDataset(Dataset):
    def __init__(self, gt_mode: str, ds_mode: str, spatial_dim: int = 128, exclusive_seq: int = None,
                 normalize: bool = True, gray_scaled: bool = False, preprocess_z: bool = False, transformer=None):
        super(PhantomSegDataset, self).__init__()
        self.img = []
        self.gt = []
        self.spatial_dim = spatial_dim
        self.normalize = normalize
        self.transformer = transformer
        self.mask_idx = torch.from_numpy(create_circular_mask(spatial_dim, spatial_dim))

        if ds_mode == 'global_label':
            warnings.warn('This mode is deprecate. Use PhantomSequentialDataset instead')
        if ds_mode != 'train' and transformer:
            raise AssertionError('Dataaugmentation is only allowed in training mode.')

        # select sequence
        if exclusive_seq is None:
            sample_n = dataset_sequence_splits(ds_mode)
        else:
            sample_n = [exclusive_seq]
            warnings.warn('Load only one sequence exclusively.')
        dir_name = f'./Raw/phantom/{spatial_dim}x{spatial_dim}'
        if not isdir(dir_name):
            raise NotADirectoryError('Check for existing spatial dimension. Current dir:' + getcwd())

        for n in tqdm(sample_n, desc='load data'):
            self.img.append(torch.load(join(dir_name, f'{n}_aligned_rgb.pt')))
            try:
                self.gt.append(torch.load(join(dir_name, f'{n}_{gt_mode}.pt')))
            except FileNotFoundError:
                raise FileNotFoundError('Check for correct, existing mode.')
        self.gt = torch.concat(self.gt, dim=0)
        self.img = torch.concat(self.img, dim=0)

        if gray_scaled:
            self.img = rgb_to_grayscale(self.img, num_output_channels=3)

        # clamp and normalize z
        if gt_mode == 'z' and preprocess_z:
            chunk_size = 16384
            for idx in tqdm(torch.split(torch.arange(len(self.gt)), chunk_size), desc='preprocess z'):
                gt = self.gt[idx].flatten(start_dim=-2, end_dim=-1)
                gt_q = torch.quantile(gt, 0.95, dim=-1, keepdim=True)
                gt = torch.minimum(gt, gt_q)
                min = gt.min(dim=-1, keepdim=True)[0]
                max = gt.max(dim=-1, keepdim=True)[0]
                gt = (gt - min) / (max - min)
                self.gt[idx] = gt.view(len(idx), 1, spatial_dim, spatial_dim)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = self.img[item]
        if self.transformer:
            img = self.transformer(img) / 255
            img[:, self.mask_idx] = 0
        else:
            img = img 
        if self.normalize:
            img = (img - MU.view(-1, 1, 1)) / SIGMA.view(-1, 1, 1)
        return img, self.gt[item]


if __name__ == '__main__':
    # calculate statistics
    p = PhantomSegDataset('blob', 'train', 128, normalize=False, gray_scaled=True)

    mu = p.img.float().mean(dim=(0, 2, 3)) / 255
    sigma = p.img.float().std(dim=(0, 2, 3)) / 255
    print('mu:', mu)
    print('sigma:', sigma)

    # multi label
    # gt = p.gt.squeeze().amax(dim=(-2, -1))
    # print('max gt:', gt.max())
    # print('min gt:,', gt.min())
    #
    # print('number_airways', torch.bincount(gt))

    # binary label
    print('binary class frequency', torch.bincount(p.gt[p.gt >= 0].to(torch.int8)))
