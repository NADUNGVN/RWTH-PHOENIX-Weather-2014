#############################################
#                                           #
# Load sequential data from PHOENIX-2014    #
# Updated with better augmentation & fixes  #
#                                           #
#############################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import cv2
import random
from skimage import io
import gzip

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tools.phoenix_cleanup import clean_phoenix_2014
from tools.indexs_list import idxs

import warnings
warnings.filterwarnings("ignore")


def collate_fn(data, fixed_padding=None, pad_index=1232):
    """Creates mini-batch tensors w/ same length sequences by performing padding."""
    
    def pad(sequences, t):
        lengths = [len(seq) for seq in sequences]

        if t == 'source':
            seq_shape = sequences[0].shape
            if fixed_padding:
                padded_seqs = torch.zeros(len(sequences), fixed_padding, seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
        elif t == 'target':
            padded_seqs = np.full((len(sequences), max(lengths)), fill_value=pad_index, dtype=np.int64)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths

    src_seqs = []
    trg_seqs = []
    right_hands = []

    for element in data:
        src_seqs.append(element['images'])
        trg_seqs.append(element['translation'])
        right_hands.append(element['right_hands'])

    src_seqs, src_lengths = pad(src_seqs, 'source')
    trg_seqs, trg_lengths = pad(trg_seqs, 'target')

    if type(right_hands[0]) != type(None):
        hand_seqs, hand_lengths = pad(right_hands, 'source')
    else:
        hand_seqs = None
        hand_lengths = None

    return src_seqs, src_lengths, trg_seqs, trg_lengths, hand_seqs, hand_lengths


class PhoenixDataset(Dataset):
    """Sequential Sign language images dataset with improved augmentation."""

    def __init__(self, csv_file, root_dir, segment_path, lookup_table, random_drop, 
                 uniform_drop, istrain, transform=None, rescale=224, sos_index=1, 
                 eos_index=2, unk_index=0, fixed_padding=None, hand_dir=None, 
                 hand_transform=None, channels=3, use_segmentation=False):

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.segment_path = segment_path
        self.hand_dir = hand_dir
        self.random_drop = random_drop
        self.uniform_drop = uniform_drop
        self.transform = transform
        self.hand_transform = hand_transform
        self.istrain = istrain
        self.rescale = rescale
        self.channels = channels
        self.use_segmentation = use_segmentation and segment_path is not None

        self.eos_index = eos_index
        self.unk_index = unk_index
        self.sos_index = sos_index

        with open(lookup_table, 'rb') as pickle_file:
            self.lookup_table = pickle.load(pickle_file)

    def __len__(self):
        return len(self.annotations)

    def _load_segmentation(self, seg_path):
        """Load segmentation mask from gzip file."""
        try:
            with gzip.open(seg_path, 'rb') as f:
                segmentation = np.load(f)
            return segmentation
        except:
            return None

    def _apply_segmentation(self, image, segmentation):
        """Apply segmentation mask to remove background."""
        if segmentation is None:
            return image
        
        # Resize segmentation to match image size
        seg_resized = cv2.resize(segmentation.astype(np.float32), 
                                  (image.shape[1], image.shape[0]))
        seg_mask = (seg_resized > 0.5).astype(np.uint8)
        
        # Apply mask (keep foreground, zero background)
        seg_3ch = np.repeat(seg_mask[..., np.newaxis], 3, axis=2)
        masked_image = image * seg_3ch
        
        return masked_image

    def __getitem__(self, idx):
        name = self.annotations.iloc[idx, 0].split('|')[0]
        seq_name = os.path.join(self.root_dir, name, '1')
        
        # Get segmentation path if using segmentation
        segments_name = None
        if self.use_segmentation and self.segment_path:
            segments_name = os.path.join(self.segment_path, name)

        for path, d, files in os.walk(seq_name):
            if self.istrain:
                indexs = idxs(len(files), random_drop=self.random_drop, uniform_drop=self.uniform_drop)
            else:
                if self.random_drop:
                    indexs = idxs(len(files), random_drop=None, uniform_drop=self.random_drop)
                else:
                    indexs = idxs(len(files), random_drop=None, uniform_drop=self.uniform_drop)
            
            seq_length = len(indexs)
            trsf_images = torch.zeros((seq_length, self.channels, self.rescale, self.rescale))

            # Random crop parameters (consistent across frames in sequence)
            w1 = random.randint(0, 32)  # 256 - 224 = 32
            h1 = random.randint(0, 32)

            # Hand images
            if self.hand_dir:
                hand_path = os.path.join(self.hand_dir, name)
                hand_images = torch.zeros((seq_length, self.channels, 112, 112))
            else:
                hand_images = None

            images = sorted(os.listdir(seq_name))
            
            for i, ind in enumerate(indexs):
                img = images[ind]
                img_name = os.path.join(seq_name, '{}{:03d}'.format(img[:-9], ind) + '-0.png')
                
                # Load image
                image = cv2.imread(img_name)
                if image is None:
                    # Fallback: try loading with different naming
                    img_name = os.path.join(seq_name, img)
                    image = cv2.imread(img_name)
                    if image is None:
                        continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply segmentation if available
                if self.use_segmentation and segments_name:
                    seg_name = os.path.join(segments_name, '{}{:03d}'.format(img[:-9], ind) + '-0.npy.gz')
                    segmentation = self._load_segmentation(seg_name)
                    if segmentation is not None:
                        image = self._apply_segmentation(image, segmentation)
                
                # Resize to 256x256 for cropping
                image = cv2.resize(image, (256, 256))
                
                # Apply crop
                if self.istrain:
                    cropped_image = image[h1:h1 + 224, w1:w1 + 224, :]
                else:
                    # Center crop for validation/test
                    cropped_image = image[16:16 + 224, 16:16 + 224, :]
                
                # Apply transforms
                if self.transform:
                    trsf_images[i] = self.transform(cropped_image)

                # Handle hand images if needed
                if self.hand_dir and hand_images is not None:
                    hand_name = os.path.join(hand_path, 'images{:04d}.png'.format(ind))
                    if os.path.exists(hand_name):
                        hand_img = io.imread(hand_name)
                        if hand_img.shape[2] == self.channels:
                            hand_images[i] = self.hand_transform(hand_img)
                        else:
                            hand_images[i] = self.hand_transform(hand_img[:, :, :self.channels])

        # Get translation
        translation = self.annotations.iloc[idx, 0].split('|')[-1]
        translation = clean_phoenix_2014(translation)
        translation = translation.split(' ')
        
        trans = []
        for word in translation:
            if word in self.lookup_table.keys():
                trans.append(self.lookup_table[word])
            else:
                trans.append(self.unk_index)

        return {'images': trsf_images, 'right_hands': hand_images, 'translation': trans}


def get_train_transforms(rescale=224):
    """Enhanced data augmentation for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        # Spatial augmentations
        transforms.RandomAffine(
            degrees=15,           # Rotation
            translate=(0.1, 0.1), # Translation
            scale=(0.9, 1.1),     # Scale
            shear=5               # Shear
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        # Color augmentations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.Resize((rescale, rescale)),
        transforms.ToTensor(),
        # Normalize with ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Random erasing for regularization
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(rescale=224):
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((rescale, rescale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_hand_transforms(istrain=False):
    """Hand region transforms."""
    if istrain:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def loader(csv_file, root_dir, segment_path, lookup, rescale, batch_size, num_workers, 
           random_drop, uniform_drop, show_sample, istrain=False, mean_path=None, 
           fixed_padding=None, hand_dir=None, data_stats=None, hand_stats=None, 
           channels=3, use_segmentation=False):
    """
    Create dataloader with improved augmentation.
    
    Args:
        use_segmentation: Whether to use segmentation masks (if available)
    """
    
    if istrain:
        trans = get_train_transforms(rescale)
        hand_trans = get_hand_transforms(istrain=True)
    else:
        trans = get_val_transforms(rescale)
        hand_trans = get_hand_transforms(istrain=False)

    transformed_dataset = PhoenixDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        segment_path=segment_path,
        lookup_table=lookup,
        random_drop=random_drop,
        uniform_drop=uniform_drop,
        transform=trans,
        rescale=rescale,
        istrain=istrain,
        hand_dir=hand_dir,
        hand_transform=hand_trans,
        channels=channels,
        use_segmentation=use_segmentation
    )

    size = len(transformed_dataset)

    dataloader = DataLoader(
        transformed_dataset, 
        batch_size=batch_size,
        shuffle=istrain,  # Only shuffle for training
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=istrain  # Drop last incomplete batch for training
    )

    if show_sample and istrain:
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, images_length, trans_batch, trans_length, _, _ = sample_batched
            grid = utils.make_grid(images_batch[0, :images_length[0]])
            grid = grid.numpy()
            img = np.transpose(grid, (1, 2, 0))
            # Denormalize for visualization
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.figure(figsize=(20, 4))
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('data_sample.png', bbox_inches='tight', dpi=150)
            plt.close()
            break

    return dataloader, size
