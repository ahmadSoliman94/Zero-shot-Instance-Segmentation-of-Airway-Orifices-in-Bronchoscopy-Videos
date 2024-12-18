from abc import ABC, abstractmethod
from pathlib import Path
from warnings import warn

import torch
from albumentations.augmentations.domain_adaptation import fourier_domain_adaptation
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm, trange
from torchvision.transforms.functional import rgb_to_grayscale

from phantom_sequential_ds import MU, SIGMA
from datasets.label import load_gt


class SequentialBaseDataset(Dataset, ABC):
    def __init__(self, label_csv_path: str, spatial_dim: int, normalize: bool = False, use_fda: bool = False,
                 gray_scaled: bool = False):
        super(SequentialBaseDataset, self).__init__()
        self.spatial_dim = spatial_dim
        self.gt, mask = load_gt(label_csv_path)
        self.img = self.load_images(mask)

        # sanity check
        assert self.img.shape[-1] == self.img.shape[-2] == self.spatial_dim
        assert len(self.gt) == len(self.img)
        if use_fda:
            src_domain = torch.load('./datasets/phantom/data/tensor_data/128x128/5_aligned_rgb.pt')[1350]
            src_domain = src_domain.div(255).permute(1, 2, 0).numpy()
            for i in trange(len(self.img), desc='performing domain adaptation'):
                adapted_img = self.img[i].permute(1, 2, 0).numpy()
                adapted_img = fourier_domain_adaptation(adapted_img, src_domain, beta=0.03)
                self.img[i] = torch.from_numpy(adapted_img).permute(2, 0, 1)

        if gray_scaled:
            for i in trange(len(self.img), desc='gray sclaing'):
                self.img[i] = rgb_to_grayscale(self.img[i], num_output_channels=3)

        if normalize:
            warn('Foreign statistics from phantom is used for normalization!')
            for img_chunk in self.img.chunk(len(self.img) // 128):
                img_chunk.sub_(MU.view(1, 3, 1, 1)).div_(SIGMA.view(1, 3, 1, 1))

    @abstractmethod
    def load_images(self, mask: torch.Tensor) -> torch.Tensor:
        pass

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, item):
        return self.img[item], self.gt[item]


class SequentialDataset(SequentialBaseDataset):
    datasets_crop_params = {
        'real': ((57, 511), (267, 643)),
        'lehr': ((61, -1), (107, 1361)),
        'vb': ((0, 492), (0, 564))
    }

    dataset_img_dirs = {
        'real': './datasets/real/data/img',
        'lehr': './datasets/lehr/data/img'
    }

    dataset_label_files = {
        'real': './datasets/real/data/Ein-Ausatmen-Labels.csv',
        'lehr': './datasets/lehr/data/Lehrvideo-Labels.txt'
    }

    def __init__(self, dataset: str, spatial_dim: int, normalize: bool = True, use_fda: bool = False,
                 gray_scaled: bool = False):
        if dataset not in self.datasets_crop_params.keys() or dataset not in self.dataset_img_dirs.keys() or dataset not in self.dataset_label_files.keys():
            raise ValueError(f'{dataset} is unknown.')
        self.img_dir = self.dataset_img_dirs[dataset]
        self.h, self.w = self.datasets_crop_params[dataset]
        super(SequentialDataset, self).__init__(self.dataset_label_files[dataset], spatial_dim, normalize, use_fda, gray_scaled)

    def load_images(self, mask: torch.Tensor) -> torch.Tensor:
        imgs_file_list = [p for i, p in enumerate(sorted(Path(self.img_dir).iterdir())) if mask[i]]
        imgs = torch.empty(len(imgs_file_list), 3, self.spatial_dim, self.spatial_dim)
        for i, img_file in enumerate(tqdm(imgs_file_list, desc='load data')):
            i_tmp = read_image(str(img_file))[:, self.h[0]: self.h[1], self.w[0]:self.w[1]]
            imgs[i] = interpolate(i_tmp.div(255).unsqueeze(0), size=self.spatial_dim, mode='bilinear')

        return imgs


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = SequentialDataset('real', 128, False, True)
    plt.imshow(ds.img[-1].permute(1, 2, 0))
    plt.show()
