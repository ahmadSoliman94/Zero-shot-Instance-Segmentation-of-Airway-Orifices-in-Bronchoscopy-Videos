import re
from csv import reader

import torch

COLUMN_NAMES = ['Time', 'Trachea', 'RMB', 'TriRUL', 'RB1', 'RB2', 'RB3', 'BronInt', 'RB4_5', 'RB4', 'RB5', 'RB6',
                'RB7', 'TriRLL',
                'RB9_10', 'RB8', 'RB9', 'RB10', 'LMB', 'TriLUL', 'LB1_2_3', 'LB1', 'LB2', 'LB3', 'LB4', 'LB5', 'LB4_5',
                'LLB6',
                'TriLLB', 'LB6', 'LB8', 'LB9', 'LB10', 'LB9_10', 'Larynx', 'MASK', 'RLL7', 'LB1_2']

# consider only labels present in phantom dataset
LABEL = []
LABEL_IDX = []
for i, l in enumerate(COLUMN_NAMES[1:]):  # skip time
    if l not in ['MASK', 'Larynx', 'LB1_2_3'] and not re.match(r"(R|L)B[0-9]+$", l):
        LABEL_IDX.append(i)
        LABEL.append(l)


def load_gt(csv_path: str):
    gt = []
    with open(csv_path, 'r') as label_cvs_file:
        csv_reader = reader(label_cvs_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            # skip time colum
            t = torch.tensor([int(e) for e in row[1:]], dtype=torch.bool)
            gt.append(t)
    gt = torch.stack(gt, dim=0)

    # exclude samples where MASK==1 (marks transition frames)
    select_mask_colum = torch.arange(gt.shape[1]) == (COLUMN_NAMES.index('MASK') - 1)
    mask = ~gt[:, select_mask_colum].squeeze()  # MASK==0
    mask = torch.bitwise_and(mask, gt[:, LABEL_IDX].sum(1) == 1)  # consider only one-hot
    gt = gt[mask][:, LABEL_IDX]  # consider only labels presents in phantom dataset

    if not all(gt.sum(1) == 1):
        raise RuntimeError('Class labels are not one hot encoded.')

    gt = gt.float().argmax(1)
    return gt, mask
