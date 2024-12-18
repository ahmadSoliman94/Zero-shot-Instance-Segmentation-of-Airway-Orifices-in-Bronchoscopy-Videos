import json
import pathlib
import warnings
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from skimage import measure
from torch import nn
from torch.nn import functional as F

from phantom_utils import create_circular_mask


def dice_coeff(prediction: torch.Tensor, labels: torch.Tensor, max_label: int):
    dice = torch.full((max_label,), -1, device=labels.device, dtype=torch.float)
    for label_num in range(1, max_label + 1):
        iflat = (prediction == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


class Object2StringEncoder(json.JSONEncoder):
    def default(self, obj):
        if not isinstance(obj, (dict, list, str, int, float, bool)):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class TorchSaver:
    def __init__(self, root_path: str, label: str):
        self.path = pathlib.Path(root_path)
        self.path = self.path.joinpath('_'.join([
            label, datetime.now().strftime('%d-%m_%H-%M')]))
        self.path.mkdir(exist_ok=True)

    def save_dict(self, d: dict, label: str):
        for k, v in d.items():
            torch.save(v, self.path.joinpath(f'{k}_{label}.pt').resolve())

    def save_model(self, m: torch.nn.Module, label: str):
        warnings.warn('This way of savin the model is deprecated! Use LoadableModel from modelio instead.')
        m.eval()
        torch.save(m, self.path.joinpath(f'model_{label}.pt').resolve())

    def save_2_data_frame(self, d: dict, label: str):
        df = pd.DataFrame()
        for k, v in d.items():
            assert isinstance(v, torch.Tensor)
            df[k] = v.cpu()
        df.insert(0, 'epoch', torch.arange(1, df.shape[0] + 1))
        df.to_pickle(str(self.path.joinpath(f'training_stats_{label}.pkl').resolve()))
        df.to_csv(str(self.path.joinpath(f'training_stats_{label}.csv').resolve()))

    def save_hyper_parameters_2_json(self, hp: SimpleNamespace):
        with open(self.path.joinpath('hyper_parameters.json'), mode='w') as f:
            f.write(json.dumps(hp.__dict__, indent=4, cls=Object2StringEncoder))


def set_required_grad(m: torch.nn.Module, status: bool):
    for p in m.parameters():
        p.requires_grad = status


class GroundTruthHandler:
    def __init__(self, gt_mode: str, max_label: int = None):
        if not gt_mode in ['blob', 'z']:
            raise NotImplementedError('Unknown ground truth mode.')
        self.max_label = max_label

        self.need_dice = gt_mode in ['blob']
        if self.need_dice and max_label is None:
            raise RuntimeError('Please specify max_label, when using dice')

        if gt_mode == 'blob':
            self.interpolation_mode = ('bilinear', 'nearest')
        elif gt_mode == 'z':
            self.interpolation_mode = ('bilinear', 'bilinear')
        else:
            raise NotImplementedError('Set interpolation mode.')

    @torch.no_grad()
    def error_fn(self, logits, gt):
        if self.need_dice:
            return dice_with_logits(logits, gt, self.max_label)
        else:
            return CircularExcludingLoss(spatial_dim=100, loss='l1')(logits, gt)


@torch.no_grad()
def dice_with_logits(logits: torch.Tensor, labels: torch.Tensor, max_label: int, threshold=0.5):
    """

    :param logits: raw predictions
    :param labels: ground truth
    :param max_label: highest class index of dataset
    :param threshold: to convert probabilities logits. If is none argmax is used over the channels
    :return:
    """
    if threshold is None:
        return dice_coeff(torch.argmax(logits, dim=1), labels, max_label)
    else:
        if logits.shape[1] > 1:
            raise NotImplementedError('have to add support BCE')
        else:
            assert labels.unique().tolist() == [0, 1]
            labels = labels.to(torch.bool)
            prediction = logits > threshold

            iflat = prediction.view(-1).float()
            tflat = labels.view(-1).float()
            intersection = torch.mean(iflat * tflat)
            dice = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
            return dice


class CircularExcludingLoss(nn.Module):
    """
        calculate the loss excluding the area outside the circular mask
    """

    def __init__(self, spatial_dim: int, loss: str):
        super(CircularExcludingLoss, self).__init__()
        loss_zoo = {
            'l1': F.l1_loss,
            'l2': F.mse_loss,
            'bce_with_logits': F.binary_cross_entropy_with_logits
        }
        self.loss_fcn = loss_zoo[loss.lower()]
        self.mask = torch.from_numpy(~create_circular_mask(spatial_dim, spatial_dim)).view(1, 1, spatial_dim,
                                                                                           spatial_dim)

    def forward(self, input, target):
        mask = self.mask.expand_as(input).to(input.device, non_blocking=True)
        masked_input = input[mask]
        masked_target = target[mask]
        loss = self.loss_fcn(masked_input, masked_target)

        return loss


def extract_airways(label_img: np.ndarray):
    """
    extract airways from segmentation mask
    :param label_img:
    :return: labeled image, centroids [#c, #dim], number of instances
    """
    assert label_img.dtype == np.bool_
    label, num = measure.label(label_img, return_num=True)
    rois = measure.regionprops(label)
    airway_centroids = []
    for region in rois:
        airway_centroids.append(region.centroid)
    airway_centroids = np.asarray(airway_centroids)

    return label, airway_centroids, num


def sample_from_image(img: torch.Tensor, points: torch.Tensor):
    """
    sample from the given img at the given pixel coordinates

    :param img: [C, H, W]
    :param points: pixel coordinates [N, 2], with (y, x) due to pytorch's dim order of [H, W]
    :return: tensor [N]
    """
    assert img.ndim == 3
    assert points.shape[1] == 2

    # map coordinates to range [-1, 1]
    points = (points.float() / (torch.as_tensor(img.shape[1:]).unsqueeze(0) - 1)) * 2 - 1
    if torch.any(torch.bitwise_and(points < -1, points > 1)):
        warnings.warn('Coordinates not in image region. Using padding.')
    grid = points.view(1, -1, 1, 2)
    sample = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True)

    return sample.squeeze()


if __name__ == '__main__':
    # test saver
    s = TorchSaver('./segmentation/results', 'test')
    s.save_2_data_frame({'loss_train': torch.rand(30), 'score_train': torch.rand(30),
                         'loss_val': torch.rand(30), 'score_val': torch.rand(30)}, 'test')
