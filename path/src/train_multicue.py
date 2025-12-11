"""
Training script for Multi-Cue CSLR Model
"""

import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
import datetime as dt
import _pickle as pickle
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from models import MultiCueCSLR, make_multicue_model
from models.multicue_model import MultiCueLoss
from dataloader_multicue import create_multicue_dataloader, collate_fn_multicue

import progressbar
from jiwer import wer
import GPUtil

# TensorFlow for CTC decoding
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

###
# Argument parsing
##############

parser = argparse.ArgumentParser(description='Training Multi-Cue CSLR Model')

# Data paths
parser.add_argument('--train_data', type=str, required=True,
                    help='Root directory for training data (pre-extracted features)')
parser.add_argument('--val_data', type=str, required=True,
                    help='Root directory for validation data')
parser.add_argument('--train_csv', type=str, required=True,
                    help='Training annotations CSV')
parser.add_argument('--val_csv', type=str, required=True,
                    help='Validation annotations CSV')
parser.add_argument('--lookup_table', type=str, default='data/label/slr_lookup.txt',
                    help='Path to gloss lookup table')

# Model config
parser.add_argument('--d_model', type=int, default=768,
                    help='Model dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention heads')
parser.add_argument('--n_temporal_layers', type=int, default=6,
                    help='Number of Conformer layers')
parser.add_argument('--n_fusion_layers', type=int, default=2,
                    help='Number of fusion layers')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--fusion_type', type=str, default='cross_attention',
                    choices=['cross_attention', 'simple', 'adaptive'],
                    help='Type of fusion module')
parser.add_argument('--temporal_type', type=str, default='conformer',
                    choices=['conformer', 'lightweight'],
                    help='Type of temporal encoder')

# Training config
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs')
parser.add_argument('--lr_backbone', type=float, default=1e-5,
                    help='Learning rate for backbones')
parser.add_argument('--lr_head', type=float, default=1e-4,
                    help='Learning rate for other layers')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='Number of warmup epochs')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='Gradient clipping value')

# Loss weights
parser.add_argument('--lambda_aux', type=float, default=0.3,
                    help='Weight for auxiliary CTC losses')
parser.add_argument('--lambda_consist', type=float, default=0.0,
                    help='Weight for consistency loss (0 = disabled, recommended)')

# Data config
parser.add_argument('--random_drop', type=float, default=0.5,
                    help='Frame drop probability')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers')
parser.add_argument('--use_raw', action='store_true',
                    help='Use raw video frames instead of pre-extracted features')

# Saving and logging
parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='Directory to save experiments')
parser.add_argument('--save_steps', type=int, default=10,
                    help='Save model every N epochs')
parser.add_argument('--valid_steps', type=int, default=1,
                    help='Validate every N epochs')

# Resume training
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained backbones')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='Random seed')
parser.add_argument('--debug', action='store_true',
                    help='Debug mode (1 epoch, small batch)')

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create experiment directory
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
experiment_path = os.path.join(args.save_dir, f'multicue_{start_date}')
os.makedirs(experiment_path, exist_ok=True)

print(f"\nExperiment: {experiment_path}")

# Save config
with open(os.path.join(experiment_path, 'config.txt'), 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

###
# Load data
##############

print("\nLoading data...")

train_loader = create_multicue_dataloader(
    csv_file=args.train_csv,
    data_root=args.train_data,
    lookup_table=args.lookup_table,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    is_train=True,
    random_drop=args.random_drop,
    use_raw=args.use_raw
)

val_loader = create_multicue_dataloader(
    csv_file=args.val_csv,
    data_root=args.val_data,
    lookup_table=args.lookup_table,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    is_train=False,
    random_drop=args.random_drop,  # Will use uniform drop for validation
    use_raw=args.use_raw
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")

# Load vocabulary
with open(args.lookup_table, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
blank_id = vocab_size - 1
vocab_inv = {v: k for k, v in vocab.items()}

print(f"Vocabulary size: {vocab_size}")

###
# Create model
##############

print("\nCreating model...")

model = make_multicue_model(
    vocab_size=vocab_size,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_temporal_layers=args.n_temporal_layers,
    n_fusion_layers=args.n_fusion_layers,
    dropout=args.dropout,
    pretrained=args.pretrained,
    fusion_type=args.fusion_type,
    temporal_type=args.temporal_type
)

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

###
# Setup optimizer and scheduler
##############

# Different learning rates for backbone and head
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if 'encoder' in name and 'backbone' in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': args.lr_backbone},
    {'params': head_params, 'lr': args.lr_head}
], weight_decay=args.weight_decay)

# Cosine annealing with warm restarts
total_steps = args.num_epochs * len(train_loader)
warmup_steps = args.warmup_epochs * len(train_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[args.lr_backbone, args.lr_head],
    total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    anneal_strategy='cos'
)

# Loss function
criterion = MultiCueLoss(
    blank_id=blank_id,
    lambda_aux=args.lambda_aux,
    lambda_consist=args.lambda_consist
)

###
# Training functions
##############

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    bar = progressbar.ProgressBar(maxval=len(dataloader))
    bar.start()
    
    for step, batch in enumerate(dataloader):
        bar.update(step)
        
        # Move to device
        frames = batch['frames'].to(device)
        hands = batch['hands'].to(device)
        hand_masks = batch['hand_masks'].to(device)
        poses = batch['poses'].to(device)
        targets = batch['translations'].to(device)
        frame_lengths = batch['frame_lengths']
        trans_lengths = batch['trans_lengths']
        
        # Forward
        optimizer.zero_grad()
        outputs = model(frames, hands, poses, frame_lengths, hand_masks)
        
        # Compute loss
        input_lengths = torch.tensor(frame_lengths, dtype=torch.long)
        target_lengths = torch.tensor(trans_lengths, dtype=torch.long)
        
        losses = criterion(outputs, targets, input_lengths, target_lengths)
        
        # Backward
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += losses['total'].item()
        num_batches += 1
        
        # Free memory
        del frames, hands, poses, targets, outputs, losses
    
    bar.finish()
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, vocab_inv):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_wer = 0
    num_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            frames = batch['frames'].to(device)
            hands = batch['hands'].to(device)
            hand_masks = batch['hand_masks'].to(device)
            poses = batch['poses'].to(device)
            targets = batch['translations'].to(device)
            frame_lengths = batch['frame_lengths']
            trans_lengths = batch['trans_lengths']
            
            # Forward
            outputs = model(frames, hands, poses, frame_lengths, hand_masks)
            
            # Compute loss
            input_lengths = torch.tensor(frame_lengths, dtype=torch.long)
            target_lengths = torch.tensor(trans_lengths, dtype=torch.long)
            
            losses = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Decode predictions
            log_probs = outputs['fused']  # (T, B, V)
            
            # CTC beam search decoding
            decodes, _ = tf.nn.ctc_beam_search_decoder(
                inputs=log_probs.cpu().numpy(),
                sequence_length=input_lengths.numpy(),
                merge_repeated=False,
                beam_width=10,
                top_paths=1
            )
            
            pred = tf.sparse.to_dense(decodes[0]).numpy()
            
            # Compute WER
            for i in range(len(targets)):
                gt_seq = targets[i, :trans_lengths[i]]
                pred_seq = pred[i]
                
                gt_text = ' '.join([vocab_inv.get(x.item(), '<unk>') for x in gt_seq])
                pred_text = ' '.join([vocab_inv.get(x, '<unk>') for x in pred_seq])
                
                total_wer += wer(gt_text, pred_text)
                num_samples += 1
    
    avg_loss = total_loss / num_batches
    avg_wer = total_wer / num_samples
    
    return avg_loss, avg_wer


###
# Training loop
##############

print("\nStarting training...")

best_wer = float('inf')
start_epoch = 0

# Resume from checkpoint
if args.resume:
    print(f"Resuming from {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_wer = checkpoint.get('best_wer', float('inf'))

# Debug mode
if args.debug:
    args.num_epochs = 1

train_losses = []
val_losses = []
val_wers = []

for epoch in range(start_epoch, args.num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{args.num_epochs}")
    print(f"LR: backbone={optimizer.param_groups[0]['lr']:.2e}, head={optimizer.param_groups[1]['lr']:.2e}")
    
    # Train
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
    train_time = time.time() - start_time
    
    train_losses.append(train_loss)
    print(f"Train Loss: {train_loss:.4f} ({train_time:.1f}s)")
    
    # Validate
    if epoch % args.valid_steps == 0:
        val_loss, val_wer_score = evaluate(model, val_loader, criterion, device, vocab_inv)
        val_losses.append(val_loss)
        val_wers.append(val_wer_score)
        
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer_score:.4f}")
        
        # Save best model
        if val_wer_score < best_wer:
            best_wer = val_wer_score
            print(f"New best WER! Saving model...")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(experiment_path, 'best_model.pt'))
        
        # Log
        log_str = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_wer={val_wer_score:.4f}, best_wer={best_wer:.4f}"
        with open(os.path.join(experiment_path, 'log.txt'), 'a') as f:
            f.write(log_str + '\n')
    
    # Save checkpoint
    if epoch % args.save_steps == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_wer': best_wer
        }, os.path.join(experiment_path, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save learning curves
    np.save(os.path.join(experiment_path, 'learning_curves.npy'), {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_wers': val_wers
    })
    
    print(GPUtil.showUtilization())

print(f"\nTraining complete! Best WER: {best_wer:.4f}")
print(f"Model saved to: {experiment_path}")
