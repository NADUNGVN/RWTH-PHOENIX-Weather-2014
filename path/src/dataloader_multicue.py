"""
Multi-Cue DataLoader for CSLR
Loads pre-extracted: frames, hands, poses
"""

import os
import torch
import pandas as pd
import numpy as np
import _pickle as pickle
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tools.phoenix_cleanup import clean_phoenix_2014
from tools.indexs_list import idxs

import warnings
warnings.filterwarnings("ignore")


class MultiCuePhoenixDataset(Dataset):
    """
    Dataset class for Multi-Cue approach.
    Loads pre-extracted: frames, hands, poses from .npy files.
    """
    
    def __init__(
        self,
        csv_file: str,
        data_root: str,
        lookup_table: str,
        frame_size: int = 224,
        hand_size: int = 112,
        random_drop: float = 0.5,
        uniform_drop: float = None,
        is_train: bool = True,
        use_augmentation: bool = True
    ):
        """
        Args:
            csv_file: Path to annotation CSV
            data_root: Root directory containing extracted multi-cue data
            lookup_table: Path to gloss lookup table
            frame_size: Target frame size
            hand_size: Target hand crop size
            random_drop: Random frame drop probability
            uniform_drop: Uniform frame drop rate
            is_train: Whether this is training set
            use_augmentation: Whether to use data augmentation
        """
        self.annotations = pd.read_csv(csv_file)
        self.data_root = data_root
        self.frame_size = frame_size
        self.hand_size = hand_size
        self.random_drop = random_drop
        self.uniform_drop = uniform_drop
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        
        with open(lookup_table, 'rb') as f:
            self.lookup_table = pickle.load(f)
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Build index of available videos
        self._build_index()
    
    def _build_index(self):
        """Build index of videos with pre-extracted features."""
        self.valid_indices = []
        
        for idx in range(len(self.annotations)):
            name = self.annotations.iloc[idx, 0].split('|')[0]
            video_path = os.path.join(self.data_root, name)
            
            # Check if all required files exist
            frames_path = os.path.join(video_path, 'frames.npy')
            poses_path = os.path.join(video_path, 'poses.npy')
            
            if os.path.exists(frames_path) and os.path.exists(poses_path):
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)}/{len(self.annotations)} videos with extracted features")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        name = self.annotations.iloc[actual_idx, 0].split('|')[0]
        video_path = os.path.join(self.data_root, name)
        
        # Load pre-extracted features
        frames = np.load(os.path.join(video_path, 'frames.npy'))
        poses = np.load(os.path.join(video_path, 'poses.npy'))
        
        # Load hands (may have None values)
        left_hands_path = os.path.join(video_path, 'left_hands.npy')
        right_hands_path = os.path.join(video_path, 'right_hands.npy')
        
        if os.path.exists(left_hands_path):
            left_hands = np.load(left_hands_path, allow_pickle=True)
        else:
            left_hands = [None] * len(frames)
            
        if os.path.exists(right_hands_path):
            right_hands = np.load(right_hands_path, allow_pickle=True)
        else:
            right_hands = [None] * len(frames)
        
        # Frame sampling
        total_frames = len(frames)
        if self.is_train:
            indices = self._sample_frames(total_frames, random=True)
        else:
            indices = self._sample_frames(total_frames, random=False)
        
        # Apply sampling
        frames = frames[indices]
        poses = poses[indices]
        left_hands = [left_hands[i] for i in indices]
        right_hands = [right_hands[i] for i in indices]
        
        # Process each modality
        frames_tensor = self._process_frames(frames)
        hands_tensor, hand_mask = self._process_hands(left_hands, right_hands)
        poses_tensor = self._process_poses(poses)
        
        # Get translation
        translation = self._get_translation(actual_idx)
        
        return {
            'frames': frames_tensor,      # (T, 3, 224, 224)
            'hands': hands_tensor,        # (T, 2, 3, 112, 112)
            'hand_mask': hand_mask,       # (T, 2)
            'poses': poses_tensor,        # (T, 75, 3)
            'translation': translation,   # List of gloss indices
            'name': name,
            'length': len(frames_tensor)
        }
    
    def _sample_frames(self, total_frames, random=True):
        """Sample frames with random or uniform drop."""
        if self.random_drop and random and self.is_train:
            # Random drop for training
            keep_prob = 1 - self.random_drop
            num_frames = max(1, int(total_frames * keep_prob))
            indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        elif self.random_drop and not self.is_train:
            # Uniform drop for validation (use same drop rate but deterministic)
            keep_prob = 1 - self.random_drop
            num_frames = max(1, int(total_frames * keep_prob))
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
        elif self.uniform_drop:
            # Uniform drop
            keep_prob = 1 - self.uniform_drop
            num_frames = max(1, int(total_frames * keep_prob))
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
        else:
            indices = list(range(total_frames))
        
        return indices
    
    def _process_frames(self, frames):
        """Process frames with augmentation."""
        T = len(frames)
        processed = torch.zeros(T, 3, self.frame_size, self.frame_size)
        
        # Random augmentation params (same for all frames)
        # NOTE: NO FLIP for sign language - it changes meaning!
        if self.use_augmentation:
            # Color jitter only (safe for sign language)
            brightness = 1 + (random.random() - 0.5) * 0.3
            contrast = 1 + (random.random() - 0.5) * 0.3
        else:
            brightness, contrast = 1.0, 1.0
        
        for t in range(T):
            frame = frames[t]  # (H, W, 3) uint8
            
            # Resize to target size directly
            if frame.shape[0] != self.frame_size or frame.shape[1] != self.frame_size:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            
            # Color augmentation only
            if self.use_augmentation:
                frame = frame.astype(np.float32)
                frame = frame * brightness
                frame = (frame - 128) * contrast + 128
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = (frame_tensor - self.mean) / self.std
            processed[t] = frame_tensor
        
        return processed
    
    def _process_hands(self, left_hands, right_hands):
        """Process hand crops, handling missing hands."""
        T = len(left_hands)
        hands = torch.zeros(T, 2, 3, self.hand_size, self.hand_size)
        hand_mask = torch.zeros(T, 2)
        
        for t in range(T):
            # Left hand
            if left_hands[t] is not None:
                hand = self._process_single_hand(left_hands[t])
                if hand is not None:
                    hands[t, 0] = hand
                    hand_mask[t, 0] = 1
            
            # Right hand
            if right_hands[t] is not None:
                hand = self._process_single_hand(right_hands[t])
                if hand is not None:
                    hands[t, 1] = hand
                    hand_mask[t, 1] = 1
        
        return hands, hand_mask
    
    def _process_single_hand(self, hand):
        """Process a single hand crop."""
        if hand is None:
            return None
        
        try:
            # Ensure correct size
            if hand.shape[0] != self.hand_size or hand.shape[1] != self.hand_size:
                hand = cv2.resize(hand, (self.hand_size, self.hand_size))
            
            # Convert to tensor and normalize
            hand_tensor = torch.from_numpy(hand).permute(2, 0, 1).float() / 255.0
            hand_tensor = (hand_tensor - self.mean) / self.std
            
            return hand_tensor
        except Exception:
            return None
    
    def _process_poses(self, poses):
        """Normalize pose keypoints."""
        poses = poses.copy().astype(np.float32)
        
        # Center normalization using shoulder midpoint (points 11, 12)
        if poses.shape[0] > 0:
            # Check if shoulders are detected
            shoulder_conf = (poses[:, 11, 2] + poses[:, 12, 2]) / 2
            valid_frames = shoulder_conf > 0.1
            
            if valid_frames.any():
                center_x = (poses[:, 11, 0] + poses[:, 12, 0]) / 2
                center_y = (poses[:, 11, 1] + poses[:, 12, 1]) / 2
                
                # Normalize coordinates relative to center
                for t in range(len(poses)):
                    if valid_frames[t]:
                        poses[t, :, 0] = (poses[t, :, 0] - center_x[t]) * 2
                        poses[t, :, 1] = (poses[t, :, 1] - center_y[t]) * 2
        
        # Zero out missing keypoints (confidence = 0)
        missing_mask = poses[:, :, 2] < 0.1
        poses[missing_mask, :2] = 0
        
        return torch.from_numpy(poses).float()
    
    def _get_translation(self, idx):
        """Get gloss translation."""
        translation = self.annotations.iloc[idx, 0].split('|')[-1]
        translation = clean_phoenix_2014(translation)
        words = translation.split(' ')
        
        indices = []
        for word in words:
            if word in self.lookup_table:
                indices.append(self.lookup_table[word])
            else:
                indices.append(0)  # UNK
        
        return indices


def collate_fn_multicue(batch, pad_index=1232):
    """Custom collate function for multi-cue data."""
    
    # Sort by sequence length (descending)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # Get max lengths
    max_frame_len = max(item['length'] for item in batch)
    max_trans_len = max(len(item['translation']) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    frames = torch.zeros(batch_size, max_frame_len, 3, 224, 224)
    hands = torch.zeros(batch_size, max_frame_len, 2, 3, 112, 112)
    hand_masks = torch.zeros(batch_size, max_frame_len, 2)
    poses = torch.zeros(batch_size, max_frame_len, 75, 3)
    translations = torch.full((batch_size, max_trans_len), fill_value=pad_index, dtype=torch.long)
    
    frame_lengths = []
    trans_lengths = []
    names = []
    
    for i, item in enumerate(batch):
        T = item['length']
        S = len(item['translation'])
        
        frames[i, :T] = item['frames']
        hands[i, :T] = item['hands']
        hand_masks[i, :T] = item['hand_mask']
        poses[i, :T] = item['poses']
        translations[i, :S] = torch.tensor(item['translation'])
        
        frame_lengths.append(T)
        trans_lengths.append(S)
        names.append(item['name'])
    
    return {
        'frames': frames,
        'hands': hands,
        'hand_masks': hand_masks,
        'poses': poses,
        'translations': translations,
        'frame_lengths': frame_lengths,
        'trans_lengths': trans_lengths,
        'names': names
    }


class RawMultiCueDataset(Dataset):
    """
    Dataset that extracts multi-cue features on-the-fly.
    Use when pre-extracted features are not available.
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        lookup_table: str,
        frame_size: int = 224,
        hand_size: int = 112,
        random_drop: float = 0.5,
        is_train: bool = True
    ):
        """
        Args:
            csv_file: Path to annotation CSV
            root_dir: Root directory containing raw video frames
            lookup_table: Path to gloss lookup table
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.frame_size = frame_size
        self.hand_size = hand_size
        self.random_drop = random_drop
        self.is_train = is_train
        
        with open(lookup_table, 'rb') as f:
            self.lookup_table = pickle.load(f)
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Try to import MediaPipe
        try:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            self.has_mediapipe = True
        except ImportError:
            self.has_mediapipe = False
            print("Warning: MediaPipe not available. Pose/hand features will be zeros.")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        name = self.annotations.iloc[idx, 0].split('|')[0]
        seq_path = os.path.join(self.root_dir, name, '1')
        
        if not os.path.exists(seq_path):
            seq_path = os.path.join(self.root_dir, name)
        
        # Load all frames
        image_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
        
        # Sample frames
        total_frames = len(image_files)
        if self.is_train and self.random_drop:
            num_keep = max(1, int(total_frames * (1 - self.random_drop)))
            indices = sorted(np.random.choice(total_frames, num_keep, replace=False))
        else:
            indices = list(range(total_frames))
        
        T = len(indices)
        frames = torch.zeros(T, 3, self.frame_size, self.frame_size)
        hands = torch.zeros(T, 2, 3, self.hand_size, self.hand_size)
        hand_mask = torch.zeros(T, 2)
        poses = torch.zeros(T, 75, 3)
        
        for t, frame_idx in enumerate(indices):
            img_path = os.path.join(seq_path, image_files[frame_idx])
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Process frame
            frame_resized = cv2.resize(image_rgb, (self.frame_size, self.frame_size))
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frames[t] = (frame_tensor - self.mean) / self.std
            
            # Extract pose and hands if MediaPipe available
            if self.has_mediapipe:
                results = self.holistic.process(image_rgb)
                
                # Pose
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        poses[t, i] = torch.tensor([lm.x, lm.y, lm.visibility])
                
                if results.left_hand_landmarks:
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):
                        poses[t, 33 + i] = torch.tensor([lm.x, lm.y, 1.0])
                    # Crop left hand
                    hand_crop = self._crop_hand(results.left_hand_landmarks, image_rgb, w, h)
                    if hand_crop is not None:
                        hands[t, 0] = hand_crop
                        hand_mask[t, 0] = 1
                
                if results.right_hand_landmarks:
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        poses[t, 54 + i] = torch.tensor([lm.x, lm.y, 1.0])
                    # Crop right hand
                    hand_crop = self._crop_hand(results.right_hand_landmarks, image_rgb, w, h)
                    if hand_crop is not None:
                        hands[t, 1] = hand_crop
                        hand_mask[t, 1] = 1
        
        # Get translation
        translation = self._get_translation(idx)
        
        return {
            'frames': frames,
            'hands': hands,
            'hand_mask': hand_mask,
            'poses': poses,
            'translation': translation,
            'name': name,
            'length': T
        }
    
    def _crop_hand(self, hand_landmarks, image, w, h):
        """Crop hand region."""
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = 0.3
        size = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        half_size = size * (1 + padding) / 2
        
        x1 = max(0, int(center_x - half_size))
        x2 = min(w, int(center_x + half_size))
        y1 = max(0, int(center_y - half_size))
        y2 = min(h, int(center_y + half_size))
        
        if x2 > x1 and y2 > y1:
            hand_crop = image[y1:y2, x1:x2]
            hand_crop = cv2.resize(hand_crop, (self.hand_size, self.hand_size))
            hand_tensor = torch.from_numpy(hand_crop).permute(2, 0, 1).float() / 255.0
            return (hand_tensor - self.mean) / self.std
        
        return None
    
    def _get_translation(self, idx):
        """Get gloss translation."""
        translation = self.annotations.iloc[idx, 0].split('|')[-1]
        translation = clean_phoenix_2014(translation)
        words = translation.split(' ')
        
        indices = []
        for word in words:
            if word in self.lookup_table:
                indices.append(self.lookup_table[word])
            else:
                indices.append(0)
        
        return indices


def create_multicue_dataloader(
    csv_file: str,
    data_root: str,
    lookup_table: str,
    batch_size: int = 2,
    num_workers: int = 4,
    is_train: bool = True,
    random_drop: float = 0.5,
    use_raw: bool = False
) -> DataLoader:
    """
    Create multi-cue dataloader.
    
    Args:
        csv_file: Path to annotation CSV
        data_root: Root directory for data
        lookup_table: Path to gloss lookup table
        batch_size: Batch size
        num_workers: Number of data loading workers
        is_train: Training mode
        random_drop: Frame drop rate
        use_raw: Use raw video frames instead of pre-extracted features
        
    Returns:
        DataLoader
    """
    if use_raw:
        dataset = RawMultiCueDataset(
            csv_file=csv_file,
            root_dir=data_root,
            lookup_table=lookup_table,
            random_drop=random_drop,
            is_train=is_train
        )
    else:
        dataset = MultiCuePhoenixDataset(
            csv_file=csv_file,
            data_root=data_root,
            lookup_table=lookup_table,
            random_drop=random_drop,
            is_train=is_train
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn_multicue,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader
