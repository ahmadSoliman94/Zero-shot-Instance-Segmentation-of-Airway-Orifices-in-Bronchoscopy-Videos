from pathlib import Path
from warnings import warn

import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale

from phantom_dataset import MU, SIGMA
from datasets.sequential_dataset import SequentialDataset


class RealSegDataset(Dataset):
    def __init__(self, spatial_dim: int = 128, transformer=None, normalize: bool = True, gray_scale: bool = False):
        super(RealSegDataset, self).__init__()
        self.transformer = transformer
        self.normalize = normalize
        if normalize:
            warn('Foreign statistics from phantom is used for normalization!')

        gt_path = Path('./Raw/real/gt')
        # load data
        self.img = []
        self.gt = []

        h, w = SequentialDataset.datasets_crop_params['real']
        for gt_file in sorted(gt_path.iterdir()):
            img_file = gt_path.parent.joinpath('img').joinpath(gt_file.name)
            i_tmp = read_image(img_file.resolve().__str__())[:, h[0]: h[1], w[0]:w[1]]
            i_tmp = interpolate(i_tmp.unsqueeze(0).float(), size=spatial_dim, mode='bilinear').to(torch.uint8)
            self.img.append(i_tmp)

            g_tmp = read_image(gt_file.resolve().__str__())
            g_tmp = interpolate(g_tmp.unsqueeze(0).float(), size=spatial_dim, mode='nearest').to(torch.uint8)
            self.gt.append(g_tmp)

        self.img = torch.cat(self.img, dim=0)
        self.gt = torch.cat(self.gt, dim=0).div(255).to(torch.bool)

        if gray_scale:
            warn('gray scaling')
            self.img = rgb_to_grayscale(self.img, num_output_channels=3)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = self.img[item]
        if self.transformer:
            img = self.transformer(img) / 255
        else:
            img = img 
        if self.normalize:
            img = (img - MU.view(-1, 1, 1)) / SIGMA.view(-1, 1, 1)
        return img, self.gt[item]


if __name__ == '__main__':
    ds = RealDataset(128)
    # ds = RealSegDataset(normalize=False)
    pass
