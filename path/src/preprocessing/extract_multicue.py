"""
Multi-Cue Extraction Script for CSLR (Fast Multiprocessing Version)
Extracts: Full frames, Hand crops, Pose keypoints from video frames
Uses MediaPipe for pose/hand detection

Usage:
    python extract_multicue.py --data_root /path/to/phoenix --output_root /path/to/output --num_workers 8
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
import signal
warnings.filterwarnings("ignore")

# Set spawn method for multiprocessing (avoid CUDA/TF issues)
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

def init_worker():
    """Ignore SIGINT in worker processes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Global extractor per worker process
_worker_extractor = None

def get_worker_extractor(frame_size=224, hand_size=112):
    """Get or create extractor for current worker."""
    global _worker_extractor
    if _worker_extractor is None:
        _worker_extractor = MultiCueExtractor(frame_size=frame_size, hand_size=hand_size, fast_mode=True)
    return _worker_extractor

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: MediaPipe not installed. Install with: pip install mediapipe")


class MultiCueExtractor:
    """Extract multi-cue features from sign language video frames."""
    
    def __init__(self, frame_size=224, hand_size=112, fast_mode=True):
        self.frame_size = frame_size
        self.hand_size = hand_size
        
        if HAS_MEDIAPIPE:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=True,  # Faster for individual frames
                model_complexity=1 if fast_mode else 2,  # 1 is faster
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.holistic = None
    
    def extract_from_video_folder(self, video_path, max_frames=None, sample_rate=1):
        """
        Extract all cues from a video folder containing frames.
        
        Args:
            video_path: Path to folder containing video frames
            max_frames: Maximum number of frames to process (None = all)
            sample_rate: Sample every N frames (1 = all, 2 = every other frame)
            
        Returns:
            dict with keys: frames, left_hands, right_hands, poses
        """
        frames = []
        left_hands = []
        right_hands = []
        poses = []
        
        if not os.path.exists(video_path):
            return None
            
        image_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(image_files) == 0:
            return None
        
        # Sample frames
        if sample_rate > 1:
            image_files = image_files[::sample_rate]
        if max_frames and len(image_files) > max_frames:
            # Uniform sampling
            indices = np.linspace(0, len(image_files)-1, max_frames, dtype=int)
            image_files = [image_files[i] for i in indices]
        
        for img_file in image_files:
            img_path = os.path.join(video_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # 1. Full frame - resize to target size
            frame_resized = cv2.resize(image_rgb, (self.frame_size, self.frame_size))
            frames.append(frame_resized)
            
            # 2. Extract pose and hands using MediaPipe
            if self.holistic is not None:
                results = self.holistic.process(image_rgb)
                pose = self._extract_pose(results)
                left_hand, right_hand = self._extract_hands(results, image_rgb, w, h)
            else:
                # Fallback: zero pose and None hands
                pose = np.zeros((75, 3), dtype=np.float32)
                left_hand = None
                right_hand = None
            
            poses.append(pose)
            left_hands.append(left_hand)
            right_hands.append(right_hand)
        
        if len(frames) == 0:
            return None
            
        return {
            'frames': np.array(frames, dtype=np.uint8),
            'left_hands': left_hands,
            'right_hands': right_hands,
            'poses': np.array(poses, dtype=np.float32)
        }
    
    def _extract_pose(self, results):
        """
        Extract 75 keypoints from MediaPipe results:
        - 33 body pose points
        - 21 left hand points
        - 21 right hand points
        
        Returns: (75, 3) array with (x, y, confidence)
        """
        pose = np.zeros((75, 3), dtype=np.float32)
        
        # Body pose (33 points)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.visibility]
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                pose[33 + i] = [lm.x, lm.y, 1.0]
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                pose[54 + i] = [lm.x, lm.y, 1.0]
        
        return pose
    
    def _extract_hands(self, results, image, w, h):
        """Crop hand regions based on landmarks."""
        left_hand = None
        right_hand = None
        
        if results.left_hand_landmarks:
            left_hand = self._crop_hand(results.left_hand_landmarks, image, w, h)
        
        if results.right_hand_landmarks:
            right_hand = self._crop_hand(results.right_hand_landmarks, image, w, h)
        
        return left_hand, right_hand
    
    def _crop_hand(self, hand_landmarks, image, w, h, padding=0.3):
        """Crop hand region with padding."""
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        half_size = size * (1 + padding) / 2
        
        x_min = max(0, int(center_x - half_size))
        x_max = min(w, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(h, int(center_y + half_size))
        
        hand_crop = image[y_min:y_max, x_min:x_max]
        
        if hand_crop.size > 0 and hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
            hand_crop = cv2.resize(hand_crop, (self.hand_size, self.hand_size))
            return hand_crop.astype(np.uint8)
        
        return None
    
    def close(self):
        """Release resources."""
        if self.holistic is not None:
            self.holistic.close()


def process_single_video(args):
    """Process a single video (for multiprocessing)."""
    video, split_path, output_path, frame_size, hand_size, skip_existing, max_frames, sample_rate = args
    
    try:
        video_path = os.path.join(split_path, video, '1')
        if not os.path.exists(video_path):
            video_path = os.path.join(split_path, video)
        
        if not os.path.isdir(video_path):
            return (video, False, "Not a directory")
        
        output_video_path = os.path.join(output_path, video)
        
        # Skip if already exists
        if skip_existing and os.path.exists(os.path.join(output_video_path, 'frames.npy')):
            return (video, True, "Skipped (exists)")
        
        # Create new extractor for this video (can't share across processes with spawn)
        extractor = MultiCueExtractor(frame_size=frame_size, hand_size=hand_size, fast_mode=True)
        
        # Extract with sampling
        cues = extractor.extract_from_video_folder(video_path, max_frames=max_frames, sample_rate=sample_rate)
        extractor.close()
        
        if cues is None:
            return (video, False, "Extraction failed")
        
        # Save
        os.makedirs(output_video_path, exist_ok=True)
        np.save(os.path.join(output_video_path, 'frames.npy'), cues['frames'])
        np.save(os.path.join(output_video_path, 'poses.npy'), cues['poses'])
        np.save(os.path.join(output_video_path, 'left_hands.npy'), 
                np.array(cues['left_hands'], dtype=object), allow_pickle=True)
        np.save(os.path.join(output_video_path, 'right_hands.npy'), 
                np.array(cues['right_hands'], dtype=object), allow_pickle=True)
        
        return (video, True, "OK")
    except Exception as e:
        return (video, False, str(e))


def process_dataset(data_root, output_root, split='train', frame_size=224, hand_size=112, 
                    num_workers=4, skip_existing=True, max_frames=None, sample_rate=1):
    """Process entire dataset split with multiprocessing."""
    split_path = os.path.join(data_root, split)
    output_path = os.path.join(output_root, split)
    
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(split_path):
        print(f"Split path not found: {split_path}")
        return
    
    videos = sorted(os.listdir(split_path))
    print(f"Found {len(videos)} videos in {split}")
    if max_frames:
        print(f"Max frames per video: {max_frames}")
    if sample_rate > 1:
        print(f"Sample rate: every {sample_rate} frames")
    
    # Prepare arguments
    args_list = [(v, split_path, output_path, frame_size, hand_size, skip_existing, max_frames, sample_rate) for v in videos]
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # Use multiprocessing
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_video, args_list),
            total=len(videos),
            desc=f'Processing {split}'
        ))
    
    for video, success, msg in results:
        if success:
            if "Skipped" in msg:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
    
    print(f"Processed: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description='Extract multi-cue features from sign language videos')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing train/dev/test splits')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for extracted features')
    parser.add_argument('--splits', type=str, default='train,dev,test',
                        help='Comma-separated list of splits to process')
    parser.add_argument('--frame_size', type=int, default=224,
                        help='Target frame size')
    parser.add_argument('--hand_size', type=int, default=112,
                        help='Target hand crop size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--no_skip', action='store_true',
                        help='Do not skip existing files (re-extract all)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames per video (None = all). Use 50-100 for faster extraction.')
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='Sample every N frames (1=all, 2=every other frame). Default: 2 for speed.')
    args = parser.parse_args()
    
    if not HAS_MEDIAPIPE:
        print("ERROR: MediaPipe is required. Install with: pip install mediapipe")
        return
    
    print(f"Using {args.num_workers} workers, sample_rate={args.sample_rate}")
    splits = args.splits.split(',')
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        process_dataset(
            args.data_root, 
            args.output_root, 
            split.strip(),
            args.frame_size,
            args.hand_size,
            num_workers=args.num_workers,
            skip_existing=not args.no_skip,
            max_frames=args.max_frames,
            sample_rate=args.sample_rate
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
