from pathlib import Path

from label import LABEL, LABEL_IDX
from phantom_dataset import *
from label import load_gt

# weights to counter label unbalance (contained over training split)
GLOBAL_AIRWAYS_CLASS_WEIGHTS = torch.tensor([5449., 1570., 84., 929., 724., 276., 243., 2516., 1065., 625.,
                                             125., 291., 364., 467., 253.])
GLOBAL_AIRWAYS_CLASS_WEIGHTS = GLOBAL_AIRWAYS_CLASS_WEIGHTS.div(GLOBAL_AIRWAYS_CLASS_WEIGHTS.sum()).pow(-1).sqrt()


class PhantomSequentialDataset(PhantomSegDataset):
    def __init__(self, ds_mode: str, spatial_dim: int = 128, exclusive_seq: int = None,
                 normalize: bool = True, transformer=None, concat_sequences: bool = True, gray_scaled: bool = False):
        # use all functionality of phantom dataset for segmentation including loading of the images
        super(PhantomSequentialDataset, self).__init__('blob', ds_mode, spatial_dim, exclusive_seq, normalize,
                                                       transformer=transformer, gray_scaled=gray_scaled)
        if exclusive_seq is None:
            sample_n = dataset_sequence_splits(ds_mode)
        else:
            sample_n = [exclusive_seq]

        # replace the ground truth with the correct global label
        self.gt_list = []
        mask_list = []
        for n in sample_n:
            p = Path('./datasets/phantom/data/global_label/csv').joinpath(f'phantom_{n}_label.csv')
            g, m = load_gt(p)
            self.gt_list.append(g)
            mask_list.append(m)
        # distinguish between concatenated and stacked sequences
        self.uses_concat_sequences = concat_sequences
        if self.uses_concat_sequences:
            self.img = self.img[torch.cat(mask_list)]
            self.gt = torch.cat(self.gt_list)
            del self.gt_list
        else:
            img_tensor = self.img
            self.img_list = []
            cum_idx = 0
            # separate into sequences
            for i, mask in enumerate(mask_list):
                self.img_list.append(img_tensor[cum_idx:cum_idx + mask.numel()][mask])
                assert len(self.img_list[i]) == len(self.gt_list[i])
                cum_idx += mask.numel()
            del self.gt, self.img

    def __getitem__(self, item):  # todo documentation!
        if self.uses_concat_sequences:
            # use normalization and
            return super(PhantomSequentialDataset, self).__getitem__(item)
        else:
            self.img = self.img_list[item]
            self.gt = self.gt_list[item]

            normalized_img = torch.empty_like(self.img, dtype=torch.float)
            for i in range(len(self.img)):
                normalized_img[i] = super(PhantomSequentialDataset, self).__getitem__(i)[0]

            return normalized_img, self.gt

    def __len__(self):  # todo documentation!
        if self.uses_concat_sequences:
            return super(PhantomSequentialDataset, self).__len__()
        else:
            return len(self.img_list)

    @classmethod
    @property
    def label_and_index(cls):
        return LABEL, LABEL_IDX


if __name__ == '__main__':
    ds = PhantomSequentialDataset('train', concat_sequences=False)
    pass
