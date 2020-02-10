from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import glob
from natsort import natsorted
import os
import sys
import random
from skimage.exposure import rescale_intensity


def init_dataset(batch_size, single_channel=False):
    if not single_channel:
        normalize_func = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # image transforms
        image_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(224),
                normalize_func
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(224),
                normalize_func,
            ]),
        }
    else:
        normalize_func = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ],
                                 std=[0.5, ])
        ])

        # image transforms
        image_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize(224),
                normalize_func
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize(224),
                normalize_func,
            ]),
        }

    # train/test dataset & loader
    datasets = {
        split: PETDataset(root='/data/DSMC_breast468/full/', split=split,
                          transform=image_transforms[split]) for split in ['train', 'test']
    }

    train_loader = DataLoader(
        datasets['train'], batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        datasets['test'], batch_size=batch_size, shuffle=False, num_workers=8)

    dataloaders = dict(zip(['train', 'test'], [train_loader, test_loader]))

    return datasets, dataloaders


def oversampling(train_data):
    # separate minority and majority classes
    negative = train_data[train_data.diagnosis == '0']
    positive = train_data[train_data.diagnosis == '1']

    # upsample minority
    pos_upsampled = resample(positive,
                             replace=True,  # sample with replacement
                             # match number in majority class
                             n_samples=len(negative),
                             random_state=27)  # reproducible results
    # combine majority and upsampled minority
    upsampled = pd.concat([negative, pos_upsampled])

    return upsampled.values.T


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


class PETDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = transform
        self.split = split

        (self.frame_dirs, self.target_vals), self.max_depth, self.min_depth = self.get_datasets(split)

    def get_datasets(self, split, minor_class='1', suffix='/*/*', seed=42):

        frame_dirs = np.array(glob.glob(self.root + suffix))
        frame_dirs = np.array([x for x in frame_dirs if Image.open(
            os.path.join(x, os.listdir(x)[0])).size[0] == 256])
        target_vals = np.array(
            [os.path.dirname(s).split('/')[-1] for s in frame_dirs])

        minor_ixs = np.where((target_vals == minor_class))[0].tolist()
        major_ixs = np.where((target_vals != minor_class))[0].tolist()

        np.random.seed(seed)

        train_minor = np.random.choice(
            minor_ixs, round(0.8*len(target_vals))//2, replace=True).tolist()
        train_major = np.random.choice(
            major_ixs, round(0.8*len(target_vals))//2, replace=False).tolist()

        minor_ixs = list(set(minor_ixs)-set(train_minor))
        major_ixs = list(set(major_ixs)-set(train_major))

        train_ixs = train_minor + train_major
        train_ixs = np.array(train_ixs)

        test_minor = np.random.choice(
            minor_ixs, round(0.2*len(target_vals))//2, replace=False).tolist()
        test_major = np.random.choice(
            major_ixs, round(0.2*len(target_vals))//2, replace=False).tolist()

        test_ixs = test_minor + test_major
        np.random.shuffle(test_ixs)

        train_frame_dirs, test_frame_dirs = frame_dirs[train_ixs], frame_dirs[test_ixs]
        train_target_vals, test_target_vals = target_vals[train_ixs], target_vals[test_ixs]

#         train_frame_dirs, test_frame_dirs, train_target_vals, test_target_vals = \
#             train_test_split(frame_dirs, target_vals, test_size=0.2, random_state=seed)#, stratify=target_vals)
#
#         # oversample train dataset
#         train_frame_dirs, train_target_vals = oversampling(
#             pd.DataFrame(np.column_stack([train_frame_dirs, train_target_vals]), columns=['filepath', 'diagnosis']))

        data = {'train': [train_frame_dirs, train_target_vals],
                'test':
                [test_frame_dirs, test_target_vals]}

        max_depth = max(list(len(os.listdir(x)) for x in frame_dirs))
        min_depth = min(list(len(os.listdir(x)) for x in frame_dirs))

        return data[split], max_depth, min_depth

    def load_frames(self, frame_root, seed, transform=None):
        filenames = list(filter(lambda x: x != 'DCMs',
                                natsorted(os.listdir(frame_root))))

        res = []
        for fn in filenames:
            fn = os.path.join(frame_root, fn)
            img = Image.open(fn).convert('L')

            random.seed(seed)

            if transform is not None:
                img = transform(img)
            else:
                img = TF.to_tensor(img)

            res.append(img.numpy())

        res = np.array(res)
        res = torch.from_numpy(res).permute(1, 0, 2, 3)

        # zero-padding
        return F.pad(res, pad=(
            0, 0,
            0, 0,
            0, self.max_depth-res.size(1)
        ))

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, ix):
        seed = random.randint(-sys.maxsize, sys.maxsize)
        data = self.load_frames(self.frame_dirs[ix], seed, self.transform)
        target = torch.tensor(int(self.target_vals[ix])).long()
        target_onehot = torch.eye(2)[target]

        image_id = os.path.basename(self.frame_dirs[ix])
        return data, target_onehot, image_id


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                   class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
