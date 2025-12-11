"""
Test script for Multi-Cue CSLR components
Run this to verify all modules work correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_encoders():
    """Test all encoder modules."""
    print("\n" + "="*50)
    print("Testing Encoders...")
    print("="*50)
    
    from models.encoders import FrameEncoder, HandEncoder, PoseEncoder
    
    batch_size = 2
    seq_len = 16
    
    # Test FrameEncoder
    print("\n1. Testing FrameEncoder...")
    try:
        frame_encoder = FrameEncoder(output_dim=768, pretrained=False)
        frames = torch.randn(batch_size, seq_len, 3, 224, 224)
        out = frame_encoder(frames)
        print(f"   Input shape: {frames.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, 768), f"Expected (2, 16, 768), got {out.shape}"
        print("   FrameEncoder: PASSED")
    except Exception as e:
        print(f"   FrameEncoder: FAILED - {e}")
        return False
    
    # Test HandEncoder
    print("\n2. Testing HandEncoder...")
    try:
        hand_encoder = HandEncoder(output_dim=768, pretrained=False)
        hands = torch.randn(batch_size, seq_len, 2, 3, 112, 112)
        hand_mask = torch.ones(batch_size, seq_len, 2)
        out = hand_encoder(hands, hand_mask)
        print(f"   Input shape: {hands.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, 768), f"Expected (2, 16, 768), got {out.shape}"
        print("   HandEncoder: PASSED")
    except Exception as e:
        print(f"   HandEncoder: FAILED - {e}")
        return False
    
    # Test PoseEncoder
    print("\n3. Testing PoseEncoder...")
    try:
        pose_encoder = PoseEncoder(output_dim=768)
        poses = torch.randn(batch_size, seq_len, 75, 3)
        out = pose_encoder(poses)
        print(f"   Input shape: {poses.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, 768), f"Expected (2, 16, 768), got {out.shape}"
        print("   PoseEncoder: PASSED")
    except Exception as e:
        print(f"   PoseEncoder: FAILED - {e}")
        return False
    
    return True


def test_fusion():
    """Test fusion modules."""
    print("\n" + "="*50)
    print("Testing Fusion Modules...")
    print("="*50)
    
    from models.fusion import CrossModalFusion, SimpleFusion, AdaptiveFusion
    
    batch_size = 2
    seq_len = 16
    d_model = 768
    
    f_frame = torch.randn(batch_size, seq_len, d_model)
    f_hand = torch.randn(batch_size, seq_len, d_model)
    f_pose = torch.randn(batch_size, seq_len, d_model)
    
    # Test CrossModalFusion
    print("\n1. Testing CrossModalFusion...")
    try:
        fusion = CrossModalFusion(d_model=768, n_heads=8)
        out, attn = fusion(f_frame, f_hand, f_pose)
        print(f"   Input shapes: frame={f_frame.shape}, hand={f_hand.shape}, pose={f_pose.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, d_model)
        print("   CrossModalFusion: PASSED")
    except Exception as e:
        print(f"   CrossModalFusion: FAILED - {e}")
        return False
    
    # Test SimpleFusion
    print("\n2. Testing SimpleFusion...")
    try:
        fusion = SimpleFusion(d_model=768)
        out, _ = fusion(f_frame, f_hand, f_pose)
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, d_model)
        print("   SimpleFusion: PASSED")
    except Exception as e:
        print(f"   SimpleFusion: FAILED - {e}")
        return False
    
    # Test AdaptiveFusion
    print("\n3. Testing AdaptiveFusion...")
    try:
        fusion = AdaptiveFusion(d_model=768)
        out, weights = fusion(f_frame, f_hand, f_pose)
        print(f"   Output shape: {out.shape}")
        print(f"   Modality weights shape: {weights['modality_weights'].shape}")
        assert out.shape == (batch_size, seq_len, d_model)
        print("   AdaptiveFusion: PASSED")
    except Exception as e:
        print(f"   AdaptiveFusion: FAILED - {e}")
        return False
    
    return True


def test_conformer():
    """Test Conformer encoder."""
    print("\n" + "="*50)
    print("Testing Conformer Encoder...")
    print("="*50)
    
    from models.conformer import ConformerEncoder, LightweightTemporalEncoder
    
    batch_size = 2
    seq_len = 16
    d_model = 768
    
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)
    
    # Test ConformerEncoder
    print("\n1. Testing ConformerEncoder...")
    try:
        encoder = ConformerEncoder(d_model=768, n_heads=8, n_layers=2, conv_kernel=15)
        out = encoder(x, mask)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, d_model)
        print("   ConformerEncoder: PASSED")
    except Exception as e:
        print(f"   ConformerEncoder: FAILED - {e}")
        return False
    
    # Test LightweightTemporalEncoder
    print("\n2. Testing LightweightTemporalEncoder...")
    try:
        encoder = LightweightTemporalEncoder(d_model=768, n_layers=2)
        out = encoder(x, mask)
        print(f"   Output shape: {out.shape}")
        assert out.shape == (batch_size, seq_len, d_model)
        print("   LightweightTemporalEncoder: PASSED")
    except Exception as e:
        print(f"   LightweightTemporalEncoder: FAILED - {e}")
        return False
    
    return True


def test_full_model():
    """Test full MultiCueCSLR model."""
    print("\n" + "="*50)
    print("Testing Full MultiCueCSLR Model...")
    print("="*50)
    
    from models import MultiCueCSLR
    from models.multicue_model import MultiCueLoss
    
    batch_size = 2
    seq_len = 16
    vocab_size = 100
    
    # Create model with smaller config for testing
    print("\n1. Creating model...")
    try:
        model = MultiCueCSLR(
            vocab_size=vocab_size,
            d_model=256,  # Smaller for testing
            n_heads=4,
            n_temporal_layers=2,
            n_fusion_layers=1,
            conformer_kernel=15,
            dropout=0.1,
            frame_pretrained=False,
            hand_pretrained=False,
            fusion_type='cross_attention',
            temporal_type='conformer'
        )
        print(f"   Model created successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"   Model creation: FAILED - {e}")
        return False
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        frames = torch.randn(batch_size, seq_len, 3, 224, 224)
        hands = torch.randn(batch_size, seq_len, 2, 3, 112, 112)
        poses = torch.randn(batch_size, seq_len, 75, 3)
        hand_mask = torch.ones(batch_size, seq_len, 2)
        frame_lengths = [seq_len, seq_len - 2]
        
        outputs = model(frames, hands, poses, frame_lengths, hand_mask)
        
        print(f"   Fused output shape: {outputs['fused'].shape}")
        print(f"   Expected: ({seq_len}, {batch_size}, {vocab_size})")
        
        assert outputs['fused'].shape == (seq_len, batch_size, vocab_size)
        
        if 'frame' in outputs:
            print(f"   Frame auxiliary output: {outputs['frame'].shape}")
            print(f"   Hand auxiliary output: {outputs['hand'].shape}")
            print(f"   Pose auxiliary output: {outputs['pose'].shape}")
        
        print("   Forward pass: PASSED")
    except Exception as e:
        print(f"   Forward pass: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    print("\n3. Testing loss computation...")
    try:
        criterion = MultiCueLoss(blank_id=vocab_size-1, lambda_aux=0.3, lambda_consist=0.1)
        
        targets = torch.randint(0, vocab_size-1, (batch_size, 5))
        input_lengths = torch.tensor(frame_lengths)
        target_lengths = torch.tensor([5, 5])
        
        losses = criterion(outputs, targets, input_lengths, target_lengths)
        
        print(f"   Total loss: {losses['total'].item():.4f}")
        print(f"   Main loss: {losses['main'].item():.4f}")
        if 'aux' in losses:
            print(f"   Aux loss: {losses['aux'].item():.4f}")
        if 'consist' in losses:
            print(f"   Consist loss: {losses['consist'].item():.4f}")
        
        print("   Loss computation: PASSED")
    except Exception as e:
        print(f"   Loss computation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    try:
        losses['total'].backward()
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if has_grad:
            print("   Gradients computed successfully")
            print("   Backward pass: PASSED")
        else:
            print("   WARNING: No gradients found!")
    except Exception as e:
        print(f"   Backward pass: FAILED - {e}")
        return False
    
    return True


def test_dataloader():
    """Test dataloader (if data exists)."""
    print("\n" + "="*50)
    print("Testing DataLoader (basic import)...")
    print("="*50)
    
    try:
        from dataloader_multicue import MultiCuePhoenixDataset, collate_fn_multicue
        print("   DataLoader imports: PASSED")
        return True
    except Exception as e:
        print(f"   DataLoader imports: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("    MULTI-CUE CSLR COMPONENT TESTS")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['encoders'] = test_encoders()
    results['fusion'] = test_fusion()
    results['conformer'] = test_conformer()
    results['dataloader'] = test_dataloader()
    results['full_model'] = test_full_model()
    
    # Summary
    print("\n" + "="*60)
    print("    TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("   ALL TESTS PASSED!")
    else:
        print("   SOME TESTS FAILED - please check errors above")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
