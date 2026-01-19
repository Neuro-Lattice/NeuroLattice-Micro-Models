"""
Standalone ResNet Model Inference Script for Hugging Face
Loads a trained pruned ResNet checkpoint and performs inference with memory and latency measurement.

This script is COMPLETELY STANDALONE - all necessary model classes are included.
After pruning and optimization, the model is just standard ResNet-18 with redefined channel counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
import argparse
import os
import copy
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# HELPER FUNCTIONS FOR MODIFYING TORCHVISION RESNET18
# ============================================================================

class CustomBasicBlock(nn.Module):
    """
    Custom BasicBlock implementation that exactly matches torchvision's forward pass.
    This ensures perfect reconstruction without relying on torchvision's internal state.
    """
    expansion = 1
    
    def __init__(self, conv1, bn1, conv2, bn2, relu, downsample=None):
        super(CustomBasicBlock, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


def _create_pruned_basic_block_from_details(block_detail):
    """
    Create a BasicBlock from detailed structure information.
    Uses exact configurations from block_details for perfect reconstruction.
    Creates a CustomBasicBlock to avoid torchvision's internal state issues.
    """
    conv1_detail = block_detail['conv1']
    conv2_detail = block_detail['conv2']
    
    # Create conv1 with exact configuration
    conv1 = nn.Conv2d(
        conv1_detail['in_channels'],
        conv1_detail['out_channels'],
        kernel_size=conv1_detail['kernel_size'],
        stride=conv1_detail['stride'],
        padding=conv1_detail['padding'],
        bias=conv1_detail['bias']
    )
    
    # Create bn1 based on type
    if block_detail['bn1_type'] == 'Identity':
        bn1 = nn.Identity()
    else:
        bn1 = nn.BatchNorm2d(conv1_detail['out_channels'])
    
    # Create conv2 with exact configuration
    conv2 = nn.Conv2d(
        conv2_detail['in_channels'],
        conv2_detail['out_channels'],
        kernel_size=conv2_detail['kernel_size'],
        stride=conv2_detail['stride'],
        padding=conv2_detail['padding'],
        bias=conv2_detail['bias']
    )
    
    # Create bn2 based on type
    if block_detail['bn2_type'] == 'Identity':
        bn2 = nn.Identity()
    else:
        bn2 = nn.BatchNorm2d(conv2_detail['out_channels'])
    
    # Create relu with correct inplace setting
    if block_detail.get('has_relu', True):
        relu = nn.ReLU(inplace=block_detail.get('relu_inplace', True))
    else:
        relu = nn.ReLU(inplace=True)
    
    # Create downsample (shortcut) from details
    downsample = None
    if block_detail.get('has_downsample', False) and block_detail.get('downsample') is not None:
        downsample_detail = block_detail['downsample']
        if downsample_detail['type'] == 'Sequential':
            conv_detail = downsample_detail['conv']
            shortcut_conv = nn.Conv2d(
                conv_detail['in_channels'],
                conv_detail['out_channels'],
                kernel_size=conv_detail['kernel_size'],
                stride=conv_detail['stride'],
                bias=conv_detail['bias']
            )
            if downsample_detail.get('second_layer') == 'Identity':
                shortcut_identity = nn.Identity()
                downsample = nn.Sequential(shortcut_conv, shortcut_identity)
            else:
                shortcut_bn = nn.BatchNorm2d(conv_detail['out_channels'])
                downsample = nn.Sequential(shortcut_conv, shortcut_bn)
    
    # Create CustomBasicBlock with all components
    block = CustomBasicBlock(conv1, bn1, conv2, bn2, relu, downsample)
    
    return block


def _create_pruned_basic_block(in_channels, out_channels, intermediate_channels=None, 
                                use_bias=False, stride=1, has_shortcut=True):
    """
    Create a BasicBlock from scratch with pruned channels.
    This matches the exact structure after optimization (unwrapped PrunedConv2d -> nn.Conv2d).
    CRITICAL: Creates blocks with correct channels from the start, not modifying after creation.
    Fallback method when block_details are not available.
    """
    from torchvision.models.resnet import BasicBlock
    
    # Create BasicBlock with dummy channels first (will be replaced)
    block = BasicBlock(in_channels, out_channels, stride)
    
    # Determine conv1 output channels
    conv1_out = intermediate_channels if intermediate_channels is not None else out_channels
    
    # Replace conv1 with correct channels
    block.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=3, stride=stride, padding=1, bias=use_bias)
    
    # Replace bn1 (Identity if BatchNorm fused, otherwise BatchNorm2d)
    if use_bias:
        block.bn1 = nn.Identity()
    else:
        block.bn1 = nn.BatchNorm2d(conv1_out)
    
    # Replace conv2 with correct channels
    block.conv2 = nn.Conv2d(conv1_out, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
    
    # Replace bn2 (Identity if BatchNorm fused, otherwise BatchNorm2d)
    if use_bias:
        block.bn2 = nn.Identity()
    else:
        block.bn2 = nn.BatchNorm2d(out_channels)
    
    # CRITICAL: Ensure relu module exists (torchvision BasicBlock uses self.relu in forward)
    if not hasattr(block, 'relu'):
        block.relu = nn.ReLU(inplace=True)
    
    # Create downsample (shortcut) if needed
    if has_shortcut and (stride != 1 or in_channels != out_channels):
        if use_bias:
            # BatchNorm fused: downsample is Sequential(conv, Identity)
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=use_bias)
            shortcut_identity = nn.Identity()
            block.downsample = nn.Sequential(shortcut_conv, shortcut_identity)
        else:
            # Standard: downsample is Sequential(conv, bn)
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            shortcut_bn = nn.BatchNorm2d(out_channels)
            block.downsample = nn.Sequential(shortcut_conv, shortcut_bn)
    else:
        block.downsample = None  # torchvision uses None for identity shortcuts
    
    return block


def _modify_resnet18_layer(model, layer_name, in_channels, out_channels, num_blocks, stride,
                           intermediate_channels=None, use_bias=False, has_shortcut=True,
                           block_details=None):
    """
    Create a ResNet18 layer with pruned channel counts.
    Uses block_details if available for exact reconstruction, otherwise falls back to inference.
    """
    layers = []
    
    # If block_details are available, use them for exact reconstruction
    if block_details is not None and len(block_details) == num_blocks:
        print(f"[INFO] Using block_details for {layer_name} (exact reconstruction)")
        for block_detail in block_details:
            block = _create_pruned_basic_block_from_details(block_detail)
            layers.append(block)
    else:
        # Fallback: Create blocks from inferred parameters
        print(f"[INFO] Using inferred parameters for {layer_name} (fallback method)")
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            block_in_channels = in_channels if i == 0 else out_channels
            block_intermediate = intermediate_channels if i == 0 else None
            block_has_shortcut = has_shortcut if i == 0 else False  # Only first block has shortcut
            
            # Create BasicBlock from scratch with correct channels
            block = _create_pruned_basic_block(
                block_in_channels, out_channels,
                intermediate_channels=block_intermediate,
                use_bias=use_bias,
                stride=block_stride,
                has_shortcut=block_has_shortcut
            )
            layers.append(block)
    
    return nn.Sequential(*layers)


def infer_channel_counts_from_state_dict(state_dict):
    """
    Infer channel counts from state_dict keys (LEGACY - for backward compatibility).
    Handles both standard nn.Conv2d and PrunedConv2d formats.
    Only used when model_structure is not in checkpoint.
    """
    channel_counts = {}
    
    # Infer from conv1
    if 'conv1.weight' in state_dict:
        channel_counts['conv1'] = state_dict['conv1.weight'].shape[0]
    elif 'conv1.conv.weight' in state_dict:
        channel_counts['conv1'] = state_dict['conv1.conv.weight'].shape[0]
    
    # Infer from layer1
    if 'layer1.0.conv1.weight' in state_dict:
        channel_counts['layer1'] = state_dict['layer1.0.conv1.weight'].shape[0]
    elif 'layer1.0.conv1.conv.weight' in state_dict:
        channel_counts['layer1'] = state_dict['layer1.0.conv1.conv.weight'].shape[0]
    
    # Infer from layer2 - USE CONV2 OUTPUT (actual layer output channels = 80)
    if 'layer2.0.conv2.weight' in state_dict:
        channel_counts['layer2'] = state_dict['layer2.0.conv2.weight'].shape[0]  # output = 80
        # Also infer intermediate channels from conv1
        if 'layer2.0.conv1.weight' in state_dict:
            conv1_out = state_dict['layer2.0.conv1.weight'].shape[0]
            if conv1_out != channel_counts['layer2']:
                channel_counts['layer2_intermediate'] = conv1_out  # 40
    elif 'layer2.1.conv2.weight' in state_dict:
        channel_counts['layer2'] = state_dict['layer2.1.conv2.weight'].shape[0]
    elif 'layer3.0.conv1.weight' in state_dict:
        channel_counts['layer2'] = state_dict['layer3.0.conv1.weight'].shape[1]  # input channels
    elif 'layer2.0.conv2.conv.weight' in state_dict:
        channel_counts['layer2'] = state_dict['layer2.0.conv2.conv.weight'].shape[0]
        if 'layer2.0.conv1.conv.weight' in state_dict:
            conv1_out = state_dict['layer2.0.conv1.conv.weight'].shape[0]
            if conv1_out != channel_counts['layer2']:
                channel_counts['layer2_intermediate'] = conv1_out
    elif 'layer3.0.conv1.conv.weight' in state_dict:
        channel_counts['layer2'] = state_dict['layer3.0.conv1.conv.weight'].shape[1]
    
    # Infer from layer3 - USE CONV2 OUTPUT (actual layer output channels = 143)
    if 'layer3.0.conv2.weight' in state_dict:
        channel_counts['layer3'] = state_dict['layer3.0.conv2.weight'].shape[0]  # output = 143
        # Also infer intermediate channels from conv1
        if 'layer3.0.conv1.weight' in state_dict:
            conv1_out = state_dict['layer3.0.conv1.weight'].shape[0]
            if conv1_out != channel_counts['layer3']:
                channel_counts['layer3_intermediate'] = conv1_out  # 71
    elif 'layer3.1.conv2.weight' in state_dict:
        channel_counts['layer3'] = state_dict['layer3.1.conv2.weight'].shape[0]
    elif 'layer4.0.conv1.weight' in state_dict:
        channel_counts['layer3'] = state_dict['layer4.0.conv1.weight'].shape[1]  # input channels
    elif 'layer3.0.conv2.conv.weight' in state_dict:
        channel_counts['layer3'] = state_dict['layer3.0.conv2.conv.weight'].shape[0]
        if 'layer3.0.conv1.conv.weight' in state_dict:
            conv1_out = state_dict['layer3.0.conv1.conv.weight'].shape[0]
            if conv1_out != channel_counts['layer3']:
                channel_counts['layer3_intermediate'] = conv1_out
    elif 'layer4.0.conv1.conv.weight' in state_dict:
        channel_counts['layer3'] = state_dict['layer4.0.conv1.conv.weight'].shape[1]
    
    # Infer from layer4 - USE CONV2 OUTPUT (actual layer output channels = 250)
    if 'layer4.0.conv2.weight' in state_dict:
        channel_counts['layer4'] = state_dict['layer4.0.conv2.weight'].shape[0]  # output = 250
        # Also infer intermediate channels from conv1
        if 'layer4.0.conv1.weight' in state_dict:
            conv1_out = state_dict['layer4.0.conv1.weight'].shape[0]
            if conv1_out != channel_counts['layer4']:
                channel_counts['layer4_intermediate'] = conv1_out  # 125
    elif 'layer4.1.conv2.weight' in state_dict:
        channel_counts['layer4'] = state_dict['layer4.1.conv2.weight'].shape[0]
    elif 'fc.weight' in state_dict:
        channel_counts['layer4'] = state_dict['fc.weight'].shape[1]  # input to fc = 250
    elif 'layer4.0.conv2.conv.weight' in state_dict:
        channel_counts['layer4'] = state_dict['layer4.0.conv2.conv.weight'].shape[0]
        if 'layer4.0.conv1.conv.weight' in state_dict:
            conv1_out = state_dict['layer4.0.conv1.conv.weight'].shape[0]
            if conv1_out != channel_counts['layer4']:
                channel_counts['layer4_intermediate'] = conv1_out
    elif 'fc.linear.weight' in state_dict:
        channel_counts['layer4'] = state_dict['fc.linear.weight'].shape[1]
    
    # Infer from fc
    if 'fc.weight' in state_dict:
        channel_counts['fc'] = state_dict['fc.weight'].shape[0]
    elif 'fc.linear.weight' in state_dict:
        channel_counts['fc'] = state_dict['fc.linear.weight'].shape[0]
    
    return channel_counts


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load ResNet model from checkpoint.
    Handles both pruned and optimized (unwrapped) models.
    """
    print(f"\n[INFO] Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and other data
    config = checkpoint.get('config', {})
    history = checkpoint.get('history', {})
    final_test_acc = checkpoint.get('final_test_acc', 0.0)
    
    state_dict = checkpoint['model_state_dict']
    
    # Check if model structure is saved in checkpoint (preferred method)
    model_structure = None
    if 'model_structure' in checkpoint:
        print("[INFO] Using saved model structure from checkpoint...")
        model_structure = checkpoint['model_structure']
        # Exclude all metadata fields, keep only channel counts
        exclude_fields = [
            'use_bias', 'shortcut_is_sequential', 'has_shortcut', 'num_blocks', 
            'model_type', 'input_channels', 'num_classes', 'shortcut_structure',
            'block_details', 'conv1_details', 'bn1_type', 'maxpool_details', 
            'fc_details', 'optimization_state', 'model_dtype', 'is_compiled',
            'data_preprocessing', 'pytorch_version', 'training_state', 'device',
            'weight_verification', 'memory_layout', 'layer_ordering', 'custom_attributes'
        ]
        channel_counts = {k: v for k, v in model_structure.items() if k not in exclude_fields}
        channel_counts['use_bias'] = model_structure.get('use_bias', False)
        print(f"[INFO] Loaded structure: {channel_counts}")
        print(f"[INFO] BatchNorm fusion: {channel_counts['use_bias']}")
        print(f"[INFO] Shortcut structure: {model_structure.get('shortcut_is_sequential', {})}")
    else:
        # Fallback: Infer channel counts from state_dict (legacy method)
        print("[INFO] No saved structure found, inferring from state_dict...")
        channel_counts = infer_channel_counts_from_state_dict(state_dict)
        
        # Detect if BatchNorm was fused (checkpoint has conv biases) - do this before creating model
        has_conv_biases = any('.bias' in key and ('conv' in key or 'downsample' in key) and 'bn' not in key for key in state_dict.keys())
        channel_counts['use_bias'] = has_conv_biases
        print(f"[INFO] Inferred channel counts: {channel_counts}")
        model_structure = {}  # Empty structure for fallback
    
    # Determine num_classes from fc layer
    num_classes = channel_counts.get('fc', 10)
    
    # Extract shortcut structure info if available
    shortcut_structure = {}
    if 'model_structure' in checkpoint:
        shortcut_structure = checkpoint['model_structure'].get('shortcut_is_sequential', {})
        channel_counts['shortcut_structure'] = shortcut_structure
        print(f"[INFO] Shortcut structure from checkpoint: {shortcut_structure}")
    
    # Create model using torchvision's ResNet18 as base (matches pipeline approach)
    # This ensures exact structure matching with the pipeline
    print("[INFO] Creating model using torchvision ResNet18 as base (matching pipeline)...")
    model = resnet18(num_classes=num_classes)
    
    # Modify conv1, bn1, maxpool for CIFAR-10 (stride=1 instead of stride=2)
    # Use saved maxpool_details if available for exact reconstruction
    use_bias = channel_counts.get('use_bias', False)
    
    # Get conv1_details if available
    conv1_details = model_structure.get('conv1_details', {}) if model_structure else {}
    if conv1_details:
        model.conv1 = nn.Conv2d(
            conv1_details.get('in_channels', 3),
            conv1_details.get('out_channels', channel_counts['conv1']),
            kernel_size=conv1_details.get('kernel_size', 7),
            stride=conv1_details.get('stride', 1),
            padding=conv1_details.get('padding', 3),
            bias=conv1_details.get('bias', use_bias)
        )
    else:
        model.conv1 = nn.Conv2d(3, channel_counts['conv1'], kernel_size=7, stride=1, padding=3, bias=use_bias)
    
    # Get bn1_type if available
    bn1_type = model_structure.get('bn1_type', 'BatchNorm2d') if model_structure else 'BatchNorm2d'
    if bn1_type == 'Identity' or use_bias:
        model.bn1 = nn.Identity()
    else:
        model.bn1 = nn.BatchNorm2d(channel_counts['conv1'])
    
    # Get maxpool_details if available for exact reconstruction
    maxpool_details = model_structure.get('maxpool_details', {}) if model_structure else {}
    if maxpool_details:
        model.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_details.get('kernel_size', 3),
            stride=maxpool_details.get('stride', 1),
            padding=maxpool_details.get('padding', 1)
        )
        print(f"[INFO] Using saved maxpool_details: kernel_size={maxpool_details.get('kernel_size')}, stride={maxpool_details.get('stride')}, padding={maxpool_details.get('padding')}")
    else:
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    # Modify layers to match pruned channel counts
    # Use block_details if available for exact reconstruction
    layer2_intermediate = channel_counts.get('layer2_intermediate', None)
    layer3_intermediate = channel_counts.get('layer3_intermediate', None)
    layer4_intermediate = channel_counts.get('layer4_intermediate', None)
    
    # Get shortcut structure info from model_structure
    has_shortcut_info = model_structure.get('has_shortcut', {})
    
    # Get block_details if available
    block_details_dict = model_structure.get('block_details', {}) if model_structure else {}
    
    # Modify each layer (use block_details for exact reconstruction if available)
    model.layer1 = _modify_resnet18_layer(
        model, 'layer1', channel_counts['conv1'], channel_counts['layer1'], 
        2, stride=1, use_bias=use_bias,
        has_shortcut=has_shortcut_info.get('layer1', True),
        block_details=block_details_dict.get('layer1', None)
    )
    model.layer2 = _modify_resnet18_layer(
        model, 'layer2', channel_counts['layer1'], channel_counts['layer2'], 
        2, stride=2, intermediate_channels=layer2_intermediate, use_bias=use_bias,
        has_shortcut=has_shortcut_info.get('layer2', True),
        block_details=block_details_dict.get('layer2', None)
    )
    model.layer3 = _modify_resnet18_layer(
        model, 'layer3', channel_counts['layer2'], channel_counts['layer3'], 
        2, stride=2, intermediate_channels=layer3_intermediate, use_bias=use_bias,
        has_shortcut=has_shortcut_info.get('layer3', True),
        block_details=block_details_dict.get('layer3', None)
    )
    model.layer4 = _modify_resnet18_layer(
        model, 'layer4', channel_counts['layer3'], channel_counts['layer4'], 
        2, stride=2, intermediate_channels=layer4_intermediate, use_bias=use_bias,
        has_shortcut=has_shortcut_info.get('layer4', True),
        block_details=block_details_dict.get('layer4', None)
    )
    
    # Modify FC layer - use fc_details if available
    if model_structure and 'fc_details' in model_structure:
        fc_details = model_structure['fc_details']
        print("[INFO] Using fc_details for exact reconstruction")
        model.fc = nn.Linear(
            fc_details['in_features'],
            fc_details['out_features'],
            bias=fc_details['bias']
        )
    else:
        # Fallback: Use inferred values
        model.fc = nn.Linear(channel_counts['layer4'], channel_counts['fc'])
    
    # Store use_bias for later reference
    model.use_bias = use_bias
    
    # Check if model was saved in FP16 (use model_structure if available, otherwise infer)
    if model_structure and 'model_dtype' in model_structure:
        checkpoint_dtype_str = model_structure['model_dtype']
        if 'float16' in checkpoint_dtype_str or 'half' in checkpoint_dtype_str.lower():
            checkpoint_dtype = torch.float16
        else:
            checkpoint_dtype = torch.float32
        print(f"[INFO] Checkpoint dtype from structure: {checkpoint_dtype}")
    else:
        # Fallback: Infer from state_dict
        sample_weight = next(iter(state_dict.values()))
        checkpoint_dtype = sample_weight.dtype
        print(f"[INFO] Checkpoint dtype (inferred): {checkpoint_dtype}")
    
    # Verify optimization state if available
    if model_structure and 'optimization_state' in model_structure:
        opt_state = model_structure['optimization_state']
        print(f"[INFO] Optimization state:")
        print(f"  Reassembled: {opt_state.get('reassembled', False)}")
        print(f"  Unwrapped: {opt_state.get('unwrapped', False)}")
        print(f"  BN Fused: {opt_state.get('bn_fused', False)}")
        print(f"  Channels Sequential: {opt_state.get('channels_sequential', False)}")
        if not opt_state.get('channels_sequential', False):
            print(f"  [WARN] Channels may not be sequential - reconstruction might be affected!")
    
    # Verify weights before loading (if weight_verification is available)
    if model_structure and 'weight_verification' in model_structure:
        weight_verification = model_structure['weight_verification']
        print(f"[INFO] Weight verification norms from checkpoint:")
        print(f"  conv1_weight_norm: {weight_verification.get('conv1_weight_norm', 'N/A')}")
        print(f"  fc_weight_norm: {weight_verification.get('fc_weight_norm', 'N/A')}")
        print(f"  fc_bias_norm: {weight_verification.get('fc_bias_norm', 'N/A')}")
    
    # Load state dict (handle both standard and pruned formats)
    # First, try to map PrunedConv2d/PrunedBatchNorm2d keys to standard format
    print("[INFO] Mapping checkpoint keys to model structure...")
    
    # DEBUG: Print sample of checkpoint keys to understand format
    print(f"[DEBUG] Sample checkpoint keys (first 20):")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"  {key}")
    
    # Detect if BatchNorm was fused (checkpoint has conv biases)
    has_conv_biases = any('.bias' in key and ('conv' in key or 'downsample' in key) and 'bn' not in key for key in state_dict.keys())
    print(f"[INFO] Detected BatchNorm fusion: {has_conv_biases} (checkpoint has conv biases)")
    
    # Update channel_counts to indicate bias usage
    channel_counts['use_bias'] = has_conv_biases
    
    mapped_state_dict = {}
    missing_keys = []
    
    for key, value in state_dict.items():
        new_key = key
        # Keep downsample as is (torchvision uses 'downsample', checkpoint also uses 'downsample')
        # No mapping needed - torchvision ResNet18 uses 'downsample' which matches checkpoint
        # Map conv.conv.weight -> conv.weight (PrunedConv2d format)
        if '.conv.weight' in key and '.conv.conv.weight' not in key:
            new_key = key.replace('.conv.weight', '.weight')
        # Map conv.conv.bias -> conv.bias
        elif '.conv.bias' in key and '.conv.conv.bias' not in key:
            new_key = key.replace('.conv.bias', '.bias')
        # Map bn.bn.weight -> bn.weight (PrunedBatchNorm2d format)
        elif '.bn.weight' in key and '.bn.bn.weight' not in key:
            new_key = key.replace('.bn.weight', '.weight')
        # Map bn.bn.bias -> bn.bias
        elif '.bn.bias' in key and '.bn.bn.bias' not in key:
            new_key = key.replace('.bn.bias', '.bias')
        # Map bn.bn.running_mean -> bn.running_mean
        elif '.bn.running_mean' in key and '.bn.bn.running_mean' not in key:
            new_key = key.replace('.bn.running_mean', '.running_mean')
        # Map bn.bn.running_var -> bn.running_var
        elif '.bn.running_var' in key and '.bn.bn.running_var' not in key:
            new_key = key.replace('.bn.running_var', '.running_var')
        # Map linear.weight -> weight (for PrunedLinear)
        elif '.linear.weight' in key and '.linear.linear.weight' not in key:
            new_key = key.replace('.linear.weight', '.weight')
        # Map linear.bias -> bias
        elif '.linear.bias' in key and '.linear.linear.bias' not in key:
            new_key = key.replace('.linear.bias', '.bias')
        
        mapped_state_dict[new_key] = value
    
    # Keep model in FP16 if checkpoint is FP16 (match pipeline behavior)
    # Pipeline evaluates in FP16, so we should too for accuracy matching
    if checkpoint_dtype == torch.float16:
        print(f"[INFO] Checkpoint is FP16, keeping model in FP16 for inference (matching pipeline)...")
        # Keep weights in FP16 (don't convert to FP32)
        # This matches how the pipeline evaluates the model
    
    # DEBUG: Check if BatchNorm keys exist in mapped dict
    bn_keys_in_checkpoint = [k for k in mapped_state_dict.keys() if 'bn' in k.lower() or 'running' in k.lower()]
    print(f"[DEBUG] BatchNorm-related keys in mapped dict ({len(bn_keys_in_checkpoint)}): {bn_keys_in_checkpoint[:10]}...")
    
    # DEBUG: Print some mapped keys to verify mapping
    print(f"[DEBUG] Sample mapped keys (first 10):")
    for i, (k, v) in enumerate(list(mapped_state_dict.items())[:10]):
        print(f"  {k}: shape {v.shape}")
    
    # Try loading with mapped keys
    try:
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
        if missing_keys:
            print(f"[WARN] Missing keys ({len(missing_keys)}): {missing_keys[:10]}..." if len(missing_keys) > 10 else f"[WARN] Missing keys: {missing_keys}")
            # Check if missing keys are BatchNorm-related
            missing_bn = [k for k in missing_keys if 'bn' in k.lower() or 'running' in k.lower()]
            if missing_bn:
                print(f"[WARN] Missing BatchNorm keys ({len(missing_bn)}): {missing_bn[:10]}...")
                print(f"[INFO] Initializing BatchNorm layers with identity transform (no-op)...")
                # Initialize BatchNorm layers to identity (no-op)
                # This assumes BatchNorm was fused into conv layers during training
                def init_bn_identity(module):
                    if isinstance(module, nn.BatchNorm2d):
                        # Identity transform: weight=1, bias=0, running_mean=0, running_var=1
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
                        nn.init.zeros_(module.running_mean)
                        nn.init.ones_(module.running_var)
                        module.eval()  # Use running stats, not batch stats
                model.apply(init_bn_identity)
                print(f"[INFO] BatchNorm layers initialized to identity")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"[WARN] Unexpected keys: {unexpected_keys}")
        
        # DEBUG: Verify some key weights loaded correctly
        print(f"[DEBUG] Verifying loaded weights:")
        if 'conv1.weight' in mapped_state_dict:
            loaded_conv1_norm = mapped_state_dict['conv1.weight'].norm().item()
            model_conv1_norm = model.conv1.weight.data.norm().item()
            print(f"  conv1.weight: checkpoint={loaded_conv1_norm:.4f}, model={model_conv1_norm:.4f}, match={abs(loaded_conv1_norm - model_conv1_norm) < 0.01}")
        # Check downsample (torchvision naming) - should be Sequential(conv, Identity) after BatchNorm fusion
        if 'layer1.0.downsample.0.weight' in mapped_state_dict:
            loaded_sc_norm = mapped_state_dict['layer1.0.downsample.0.weight'].norm().item()
            if hasattr(model.layer1[0], 'downsample') and model.layer1[0].downsample is not None:
                if isinstance(model.layer1[0].downsample, nn.Sequential) and len(model.layer1[0].downsample) > 0:
                    model_sc_norm = model.layer1[0].downsample[0].weight.data.norm().item()
                else:
                    model_sc_norm = 0.0
            else:
                model_sc_norm = 0.0
            print(f"  layer1.0.downsample.0.weight: checkpoint={loaded_sc_norm:.4f}, model={model_sc_norm:.4f}, match={abs(loaded_sc_norm - model_sc_norm) < 0.01}")
        
        print("[INFO] Loaded state_dict successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load state_dict: {e}")
        raise
    
    # Check if model is compiled (has _orig_mod attribute)
    is_compiled = hasattr(model, '_orig_mod')
    if is_compiled:
        print(f"[INFO] Model is compiled, unwrapping for compatibility...")
        # Unwrap compiled model to avoid compatibility issues
        model = model._orig_mod
    
    # Keep model in FP16 if checkpoint is FP16 (match pipeline behavior)
    # Pipeline evaluates in FP16, so we should too for accuracy matching
    # However, if FP16 fails due to GPU/CUDA incompatibility, fall back to FP32
    if checkpoint_dtype == torch.float16:
        model = model.half()  # Convert to FP16 to match pipeline
        print(f"[INFO] Model kept in FP16 for inference (matching pipeline)")
    else:
        model = model.float()  # Otherwise use FP32
    
    model = model.to(device)
    model.eval()
    
    # Verify model structure matches saved structure
    print(f"\n[DEBUG] Model structure verification:")
    print(f"  conv1: {model.conv1.out_channels} channels (expected: {channel_counts.get('conv1', 'N/A')})")
    print(f"  layer1: {model.layer1[0].conv1.out_channels} channels (expected: {channel_counts.get('layer1', 'N/A')})")
    print(f"  layer2: {model.layer2[0].conv2.out_channels} channels (expected: {channel_counts.get('layer2', 'N/A')})")
    print(f"  layer3: {model.layer3[0].conv2.out_channels} channels (expected: {channel_counts.get('layer3', 'N/A')})")
    print(f"  layer4: {model.layer4[0].conv2.out_channels} channels (expected: {channel_counts.get('layer4', 'N/A')})")
    print(f"  fc: {model.fc.out_features} classes (expected: {channel_counts.get('fc', 'N/A')})")
    print(f"  BatchNorm fusion: {model.use_bias} (expected: {channel_counts.get('use_bias', False)})")
    
    # Test forward pass with a dummy input to verify model works
    print(f"\n[DEBUG] Testing forward pass...")
    test_input = torch.randn(1, 3, 32, 32).to(device)
    # Match input dtype to model dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        test_input = test_input.half()
    
    # Try FP16 forward pass, fall back to FP32 if it fails
    fp16_failed = False
    with torch.no_grad():
        try:
            test_output = model(test_input)
        except RuntimeError as e:
            if 'CUBLAS_STATUS_NOT_SUPPORTED' in str(e) or 'cublas' in str(e).lower():
                print(f"[WARNING] FP16 inference failed with CUDA error: {e}")
                print(f"[INFO] Falling back to FP32 for compatibility...")
                # Convert model and input to FP32
                model = model.float()
                test_input = test_input.float()
                fp16_failed = True
                # Retry forward pass
                test_output = model(test_input)
            else:
                raise
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test output shape: {test_output.shape}")
        print(f"  Test output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
        print(f"  Test output mean/std: {test_output.mean().item():.4f} / {test_output.std().item():.4f}")
        print(f"  Test output argmax: {test_output.argmax(dim=1).item()}")
        if fp16_failed:
            print(f"  [NOTE] Model is running in FP32 due to FP16 compatibility issues")
        # Check if output has reasonable values (not all zeros or NaNs)
        if torch.isnan(test_output).any():
            print(f"  ⚠ WARNING: Output contains NaN values!")
        if (test_output == 0).all():
            print(f"  ⚠ WARNING: Output is all zeros!")
        if test_output.std().item() < 0.01:
            print(f"  ⚠ WARNING: Output has very low variance (std={test_output.std().item():.6f})")
    
    print(f"\n[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Test Accuracy (from checkpoint): {final_test_acc:.2f}%")
    
    # DEBUG: Model verification
    print(f"\n[DEBUG] Model verification:")
    print(f"  Model type: {type(model)}")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  FC layer weight shape: {model.fc.weight.shape}")
    print(f"  FC layer bias shape: {model.fc.bias.shape}")
    
    # Check if FC weights are actually loaded (not just initialized)
    fc_weight_norm = model.fc.weight.data.norm().item()
    fc_bias_norm = model.fc.bias.data.norm().item()
    print(f"  FC weight norm: {fc_weight_norm:.4f} (should be > 0)")
    print(f"  FC bias norm: {fc_bias_norm:.4f} (should be > 0)")
    
    # Check a few conv layers
    print(f"  Conv1 weight norm: {model.conv1.weight.data.norm().item():.4f}")
    print(f"  Layer1.0.conv1 weight norm: {model.layer1[0].conv1.weight.data.norm().item():.4f}")
    
    # Test with a dummy input
    dummy = torch.randn(1, 3, 32, 32).to(device)
    # Match input dtype to model dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        dummy = dummy.half()
    model.eval()
    with torch.no_grad():
        # Match pipeline's forward call exactly
        try:
            test_output = model(dummy, training=False, return_masks=False, return_saliency=False)
        except TypeError:
            try:
                test_output = model(dummy, training=False)
            except TypeError:
                test_output = model(dummy)
        # Handle tuple output
        if isinstance(test_output, tuple):
            test_output = test_output[0]
        print(f"  Test output shape: {test_output.shape}")
        print(f"  Test output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
        print(f"  Test output mean/std: {test_output.mean().item():.4f} / {test_output.std().item():.4f}")
        print(f"  Test output argmax: {test_output.argmax(dim=1).item()}")
    
    # Verify forward pass against saved forward_pass_verification if available
    if model_structure and 'forward_pass_verification' in model_structure:
        fwd_verification = model_structure['forward_pass_verification']
        if 'sample_input_seed' in fwd_verification and 'sample_output' in fwd_verification:
            print(f"\n[INFO] Verifying forward pass against saved checkpoint data...")
            # Recreate the same input using the saved seed
            torch.manual_seed(fwd_verification['sample_input_seed'])
            verification_input = torch.randn(1, 3, 32, 32).to(device)
            if model_dtype == torch.float16:
                verification_input = verification_input.half()
            
            model.eval()
            with torch.no_grad():
                try:
                    verification_output = model(verification_input, training=False, return_masks=False, return_saliency=False)
                except TypeError:
                    try:
                        verification_output = model(verification_input, training=False)
                    except TypeError:
                        verification_output = model(verification_input)
                if isinstance(verification_output, tuple):
                    verification_output = verification_output[0]
            
            saved_output_stats = fwd_verification['sample_output']
            actual_output_stats = {
                'mean': float(verification_output.mean().item()),
                'std': float(verification_output.std().item()),
                'min': float(verification_output.min().item()),
                'max': float(verification_output.max().item()),
                'norm': float(verification_output.norm().item()),
                'argmax': int(verification_output.argmax().item()),
            }
            
            print(f"  Saved output stats: mean={saved_output_stats['mean']:.6f}, std={saved_output_stats['std']:.6f}, argmax={saved_output_stats['argmax']}")
            print(f"  Actual output stats: mean={actual_output_stats['mean']:.6f}, std={actual_output_stats['std']:.6f}, argmax={actual_output_stats['argmax']}")
            
            # Compare values
            mean_match = abs(saved_output_stats['mean'] - actual_output_stats['mean']) < 0.1
            std_match = abs(saved_output_stats['std'] - actual_output_stats['std']) < 0.1
            argmax_match = saved_output_stats['argmax'] == actual_output_stats['argmax']
            
            print(f"  Mean match: {mean_match} (diff: {abs(saved_output_stats['mean'] - actual_output_stats['mean']):.6f})")
            print(f"  Std match: {std_match} (diff: {abs(saved_output_stats['std'] - actual_output_stats['std']):.6f})")
            print(f"  Argmax match: {argmax_match}")
            
            if not (mean_match and std_match and argmax_match):
                print(f"  [WARN] Forward pass verification FAILED! Model output doesn't match saved checkpoint.")
                print(f"  [WARN] This indicates a structural mismatch in model reconstruction.")
            else:
                print(f"  [OK] Forward pass verification PASSED!")
            
            # Compare actual output values if available
            if 'sample_output_values' in fwd_verification:
                saved_values = fwd_verification['sample_output_values']
                actual_values = verification_output[0].cpu().tolist()
                print(f"  Comparing output values:")
                max_diff = max(abs(s - a) for s, a in zip(saved_values, actual_values))
                print(f"    Max difference: {max_diff:.6f}")
                if max_diff > 0.1:
                    print(f"    [WARN] Large difference in output values! First 5:")
                    for i in range(min(5, len(saved_values))):
                        print(f"      Class {i}: saved={saved_values[i]:.6f}, actual={actual_values[i]:.6f}, diff={abs(saved_values[i] - actual_values[i]):.6f}")
            
            # Compare layer activations if available (pinpoint where divergence occurs)
            if 'layer_activations' in model_structure:
                print(f"\n[INFO] Comparing layer activations to pinpoint divergence...")
                saved_activations = model_structure['layer_activations']
                actual_activations = {}
                
                def get_activation_hook(name):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            output = output[0]
                        actual_activations[name] = {
                            'mean': float(output.mean().item()),
                            'std': float(output.std().item()),
                            'norm': float(output.norm().item()),
                            'shape': list(output.shape),
                        }
                    return hook
                
                hooks = []
                try:
                    # Register hook on maxpool output (not conv1) to match saved checkpoint
                    # The saved checkpoint shows conv1 output as [1, 64, 16, 16], which is after maxpool
                    hooks.append(model.maxpool.register_forward_hook(get_activation_hook('conv1')))
                    # Only register bn1 hook if it's not Identity (BatchNorm not fused)
                    if not isinstance(model.bn1, nn.Identity):
                        hooks.append(model.bn1.register_forward_hook(get_activation_hook('bn1')))
                    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                        if hasattr(model, layer_name):
                            layer = getattr(model, layer_name)
                            if len(layer) > 0:
                                hooks.append(layer[0].conv2.register_forward_hook(get_activation_hook(f'{layer_name}.0.conv2')))
                    hooks.append(model.fc.register_forward_hook(get_activation_hook('fc')))
                    
                    # Run forward pass again with hooks
                    _ = model(verification_input)
                except Exception as e:
                    print(f"  [WARN] Could not capture layer activations: {e}")
                    import traceback
                    traceback.print_exc()
                    # Remove any hooks that were registered
                    for hook in hooks:
                        try:
                            hook.remove()
                        except:
                            pass
                    hooks = []
                
                # Compare activations
                print(f"  Layer-by-layer comparison:")
                for layer_name in ['conv1', 'bn1', 'layer1.0.conv2', 'layer2.0.conv2', 'layer3.0.conv2', 'layer4.0.conv2', 'fc']:
                    if layer_name in saved_activations and layer_name in actual_activations:
                        saved = saved_activations[layer_name]
                        actual = actual_activations[layer_name]
                        mean_match = abs(saved['mean'] - actual['mean']) < 0.01
                        std_match = abs(saved['std'] - actual['std']) < 0.01
                        norm_match = abs(saved['norm'] - actual['norm']) < 0.1
                        shape_match = saved['shape'] == actual['shape']
                        
                        status = "✓" if (mean_match and std_match and norm_match and shape_match) else "✗"
                        print(f"    {status} {layer_name}:")
                        print(f"      Mean: saved={saved['mean']:.6f}, actual={actual['mean']:.6f}, diff={abs(saved['mean'] - actual['mean']):.6f} {'✓' if mean_match else '✗'}")
                        print(f"      Std:  saved={saved['std']:.6f}, actual={actual['std']:.6f}, diff={abs(saved['std'] - actual['std']):.6f} {'✓' if std_match else '✗'}")
                        print(f"      Norm: saved={saved['norm']:.6f}, actual={actual['norm']:.6f}, diff={abs(saved['norm'] - actual['norm']):.6f} {'✓' if norm_match else '✗'}")
                        print(f"      Shape: saved={saved['shape']}, actual={actual['shape']} {'✓' if shape_match else '✗'}")
                        
                        if not (mean_match and std_match and norm_match):
                            print(f"      [WARN] {layer_name} activations don't match! This is where divergence starts.")
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
    
    # Include data preprocessing info in checkpoint_data for use in evaluation
    checkpoint_data = {
        'config': config,
        'history': history,
        'final_test_acc': final_test_acc,
        'channel_counts': channel_counts,
        'num_classes': num_classes,
        'data_preprocessing': model_structure.get('data_preprocessing', None) if model_structure else None,
        'model_structure': model_structure  # Include full structure for reference
    }
    
    return model, checkpoint_data


def clear_memory(device):
    """Clear GPU memory and reset statistics."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()


@torch.no_grad()
def identify_peak_concurrent_layers_correct(model, sample_input, device):
    """
    Track ACTUAL PyTorch memory usage during forward pass.
    Simplified version for standard ResNet models (not gated).
    
    Args:
        model: The ResNet model to analyze
        sample_input: Sample input tensor
        device: Device to run on
    
    Returns:
        dict with peak memory information and contributing layers (actual)
    """
    model.eval()
    
    # Clear any existing memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Track actual memory at each step
    actual_memory_at_step = {}  # step -> actual_memory_mb
    max_memory_at_step = {}  # step -> max_memory_allocated (MB)
    execution_order = []
    layer_names_at_step = {}  # step -> layer_name
    hooks = []
    step_counter = [0]  # Use list to allow modification in closure
    
    def get_memory_tracking_hook(layer_name, track_input=False):
        def hook(module, input, output):
            # Track actual GPU memory at this point using memory_allocated()
            # This shows actual memory at each step (not cumulative max)
            if device.type == 'cuda':
                actual_memory_bytes = torch.cuda.memory_allocated()
                actual_memory_mb = actual_memory_bytes / (1024 ** 2)
                actual_memory_at_step[step_counter[0]] = actual_memory_mb
                max_memory_at_step[step_counter[0]] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                layer_names_at_step[step_counter[0]] = layer_name
                execution_order.append(layer_name)
            step_counter[0] += 1
        return hook
    
    # Register hooks on all key layers to track actual memory
    if hasattr(model, 'conv1'):
        hooks.append(model.conv1.register_forward_hook(get_memory_tracking_hook('conv1', track_input=False)))
    if hasattr(model, 'bn1') and not isinstance(model.bn1, nn.Identity):
        hooks.append(model.bn1.register_forward_hook(get_memory_tracking_hook('bn1', track_input=False)))
    if hasattr(model, 'maxpool'):
        hooks.append(model.maxpool.register_forward_hook(get_memory_tracking_hook('maxpool', track_input=False)))
    
    # Register hooks for each layer and block
    for layer_idx in range(1, 5):
        layer_name = f'layer{layer_idx}'
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for block_idx in range(len(layer)):
                block = layer[block_idx]
                if hasattr(block, 'conv1'):
                    hooks.append(block.conv1.register_forward_hook(
                        get_memory_tracking_hook(f'{layer_name}.{block_idx}.conv1', track_input=False)))
                if hasattr(block, 'bn1') and not isinstance(block.bn1, nn.Identity):
                    hooks.append(block.bn1.register_forward_hook(
                        get_memory_tracking_hook(f'{layer_name}.{block_idx}.bn1', track_input=False)))
                if hasattr(block, 'conv2'):
                    hooks.append(block.conv2.register_forward_hook(
                        get_memory_tracking_hook(f'{layer_name}.{block_idx}.conv2', track_input=False)))
                if hasattr(block, 'bn2') and not isinstance(block.bn2, nn.Identity):
                    hooks.append(block.bn2.register_forward_hook(
                        get_memory_tracking_hook(f'{layer_name}.{block_idx}.bn2', track_input=False)))
                if hasattr(block, 'downsample') and block.downsample is not None:
                    hooks.append(block.downsample.register_forward_hook(
                        get_memory_tracking_hook(f'{layer_name}.{block_idx}.downsample', track_input=False)))
    
    if hasattr(model, 'avgpool'):
        hooks.append(model.avgpool.register_forward_hook(get_memory_tracking_hook('avgpool', track_input=False)))
    if hasattr(model, 'fc'):
        hooks.append(model.fc.register_forward_hook(get_memory_tracking_hook('fc', track_input=False)))
    
    # Reset peak memory stats before forward pass to match inference comparison
    # This ensures max_memory_allocated() tracks peak during this single forward pass
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Run forward pass to track actual memory
    with torch.no_grad():
        _ = model(sample_input.to(device))
    
    # Get the final max_memory_allocated() to match inference comparison
    # This ensures peak value matches what inference comparison shows
    if device.type == 'cuda':
        final_max_memory_bytes = torch.cuda.max_memory_allocated()
        final_max_memory_mb = final_max_memory_bytes / (1024 ** 2)
    else:
        final_max_memory_mb = 0

    # Find first step where max_memory_allocated reached the final peak
    # -> peak occurred *during* the layer at that step (between previous hook and this one)
    sorted_steps = sorted(actual_memory_at_step.keys())
    peak_step = sorted_steps[-1] if sorted_steps else 0
    for s in sorted_steps:
        if max_memory_at_step.get(s, 0) >= final_max_memory_mb - 0.01:
            peak_step = s
            break

    peak_during_layer = layer_names_at_step.get(peak_step, "unknown")
    if peak_step == 0:
        peak_between_hooks = f"between start of forward and post-hook of {peak_during_layer}"
    else:
        prev_layer = layer_names_at_step.get(peak_step - 1, "?")
        peak_between_hooks = f"between post-hook of {prev_layer} and post-hook of {peak_during_layer}"

    # Which layers were active at that moment (heuristic: current + inputs + same-block + block input)
    active_layers_set = {peak_during_layer}
    if peak_step > 0:
        active_layers_set.add(layer_names_at_step.get(peak_step - 1, ""))
    for suffix in (".conv2", ".bn2", ".downsample"):
        if peak_during_layer.endswith(suffix):
            block = peak_during_layer.rsplit(".", 1)[0]
            for name in (f"{block}.conv1", f"{block}.bn1"):
                if name in execution_order:
                    active_layers_set.add(name)
            block_conv1 = f"{block}.conv1"
            idx = next((i for i, n in enumerate(execution_order) if n == block_conv1), None)
            if idx is not None and idx > 0:
                active_layers_set.add(execution_order[idx - 1])
            break

    active_layers = sorted(active_layers_set - {""})

    # Build memory timeline from actual measurements
    memory_timeline = []
    peak_timestamp = peak_step
    peak_active_layers = set(active_layers)

    # Process actual memory measurements (total memory, not just activations)
    for step in sorted(actual_memory_at_step.keys()):
        actual_memory_mb = actual_memory_at_step[step]
        # Use total memory (weights + activations + overhead) to match inference comparison
        total_memory_mb = actual_memory_mb

        layer_name = layer_names_at_step.get(step, f'step_{step}')
        memory_timeline.append((step, total_memory_mb, {layer_name}))

    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Use final_max_memory_mb for peak to match inference comparison
    # But keep the timeline showing actual memory at each step
    peak_memory_mb = final_max_memory_mb
    peak_concurrent_layers = peak_active_layers
    
    # Create layer memory breakdown at peak (simplified - actual memory is total)
    layer_memory_breakdown = {
        layer: peak_memory_mb / len(peak_active_layers) if peak_active_layers else 0
        for layer in peak_active_layers
    }
    
    # Store execution order and layer info
    all_tensor_sizes = {}  # Not used for actual tracking
    
    return {
        'peak_concurrent_layers': peak_concurrent_layers,
        'peak_concurrent_memory_mb': peak_memory_mb,
        'peak_step': peak_timestamp,
        'peak_during_layer': peak_during_layer,
        'peak_between_hooks': peak_between_hooks,
        'active_layers': active_layers,
        'layer_memory_breakdown': layer_memory_breakdown,
        'memory_timeline': memory_timeline,
        'execution_order': execution_order,
        'all_tensor_sizes': all_tensor_sizes
    }


def plot_forward_pass_memory_timeline(model, sample_input, device, save_path=None, baseline_model=None):
    """
    Plot actual PyTorch memory usage during forward pass.
    
    Creates a plot showing:
    - X-axis: Forward pass execution steps
    - Y-axis: Actual memory consumption (MB)
    - Shows peak concurrent memory and when it occurs
    - If baseline_model is provided, overlays baseline (Standard ResNet-18) timeline for comparison
    
    Args:
        model: The ResNet model to analyze (pruned model)
        sample_input: Sample input tensor
        device: Device to run on
        save_path: Optional path to save plot
        baseline_model: Optional standard ResNet-18 model to overlay for comparison
    
    Returns:
        Path to saved plot
    """
    has_baseline = baseline_model is not None
    if has_baseline:
        print(f"\n[INFO] Generating forward pass memory timeline plot (Baseline vs Pruned)...")
    else:
        print(f"\n[INFO] Generating forward pass memory timeline plot (Total PyTorch Memory)...")
    
    # Get baseline timeline if provided
    result_baseline = None
    if has_baseline:
        baseline_model.eval()
        result_baseline = identify_peak_concurrent_layers_correct(baseline_model, sample_input, device)
        baseline_model = baseline_model.cpu()
        clear_memory(device)
    
    # Get pruned/main model timeline
    result = identify_peak_concurrent_layers_correct(model, sample_input, device)
    
    memory_timeline = result['memory_timeline']
    execution_order = result['execution_order']
    peak_step = result['peak_step']
    peak_memory_mb = result['peak_concurrent_memory_mb']
    
    # Extract actual memory values for pruned model
    total_memory_over_time = []
    steps = []
    for step, (step_idx, total_mb, active_layers_set) in enumerate(memory_timeline):
        steps.append(step)
        if isinstance(total_mb, (int, float)):
            total_memory_over_time.append(total_mb)
        else:
            total_memory_over_time.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    if has_baseline and result_baseline is not None:
        # Extract baseline memory values
        baseline_memory_over_time = []
        baseline_steps = []
        for step, (step_idx, total_mb, _) in enumerate(result_baseline['memory_timeline']):
            baseline_steps.append(step)
            baseline_memory_over_time.append(total_mb if isinstance(total_mb, (int, float)) else 0)
        baseline_peak_mb = result_baseline['peak_concurrent_memory_mb']
        baseline_peak_step = result_baseline['peak_step']
        
        # Plot baseline (gray) and pruned (green)
        ax.plot(baseline_steps, baseline_memory_over_time, '-', color='#555555', linewidth=2.5,
                label=f'Baseline (Standard ResNet-18)  peak={baseline_peak_mb:.2f} MB', marker='o', markersize=3, alpha=0.8)
        ax.plot(steps, total_memory_over_time, '-', color='green', linewidth=2.5,
                label=f'Pruned Model  peak={peak_memory_mb:.2f} MB', marker='o', markersize=3, alpha=0.8)
        # Peak indicators: baseline (gray), pruned (green)
        ax.axvline(x=baseline_peak_step, color='#555555', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axhline(y=baseline_peak_mb, color='#555555', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=peak_step, color='green', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=peak_memory_mb, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
        
        title = (f'Baseline vs Pruned: Total PyTorch Memory During Forward Pass\n'
                 f'Batch Size: {sample_input.size(0)}')
        info_lines = [
            f"Baseline: peak {baseline_peak_mb:.2f} MB at step {baseline_peak_step} (during {result_baseline.get('peak_during_layer', '?')})",
            f"Pruned: peak {peak_memory_mb:.2f} MB at step {peak_step} (during {result.get('peak_during_layer', '?')})",
            f"Pruned peak between: {result.get('peak_between_hooks', '')}",
        ]
        active = result.get('active_layers', [])
        if active:
            info_lines.append("Pruned active at peak: " + ", ".join(active[:8]) + ("..." if len(active) > 8 else ""))
    else:
        # Single model: pruned only
        ax.plot(steps, total_memory_over_time, 'b-', linewidth=2.5,
                label='Total PyTorch Memory (Weights + Activations + Overhead)', marker='o', markersize=4, alpha=0.8)
        ax.axvline(x=peak_step, color='red', linestyle='--', linewidth=2, label=f'Peak at Step {peak_step}', alpha=0.8)
        ax.axhline(y=peak_memory_mb, color='red', linestyle=':', linewidth=1.5, label=f'Peak: {peak_memory_mb:.2f} MB', alpha=0.6)
        ax.fill_between(steps, total_memory_over_time, alpha=0.3, color='blue')
        
        title = (f'Total PyTorch Memory During Forward Pass (Weights + Activations + Overhead)\n'
                 f'Peak: {peak_memory_mb:.2f} MB during {result.get("peak_during_layer", "?")} | '
                 f'Batch Size: {sample_input.size(0)}')
        peak_between = result.get('peak_between_hooks', '')
        active = result.get('active_layers', [])
        info_lines = [f"Peak between: {peak_between}"]
        if active:
            info_lines.append("Active at peak: " + ", ".join(active[:10]) + ("..." if len(active) > 10 else ""))
    
    ax.set_xlabel('Forward Pass Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total GPU Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    max_step = (max(len(steps) - 1, len(result_baseline['memory_timeline']) - 1)
                if (has_baseline and result_baseline is not None)
                else (len(steps) - 1 if steps else 0))
    ax.set_xlim(0, max(0, max_step))
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'forward_pass_memory_timeline_{timestamp}.png'
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved forward pass memory timeline plot to: {os.path.abspath(save_path)}")
    
    # Print summary
    print(f"\n[SUMMARY]")
    if has_baseline and result_baseline is not None:
        print(f"  Baseline: peak {result_baseline['peak_concurrent_memory_mb']:.2f} MB at step {result_baseline['peak_step']} (during {result_baseline.get('peak_during_layer', '?')})")
    print(f"  Pruned: peak {peak_memory_mb:.2f} MB at step {peak_step} (during {result.get('peak_during_layer', '?')})")
    print(f"  Total steps: {len(steps)}")
    print(f"  Peak between hooks: {result.get('peak_between_hooks', '')}")
    print(f"  Active layers at peak: {result.get('active_layers', [])}")

    return save_path


def measure_detailed_memory_breakdown(model, sample_input, device):
    """
    Measure detailed memory breakdown: weights vs activations (layer-wise).
    
    Returns:
        Dictionary with weight_memory_mb, activation_memory_mb, layer_breakdown
    """
    model.eval()
    
    x = sample_input.to(device)
    model_dtype = next(model.parameters()).dtype
    is_fp16 = model_dtype == torch.float16
    
    if is_fp16:
        x = x.half()
    
    # Clear memory
    clear_memory(device)
    
    # 1. Calculate weight memory
    weight_memory_bytes = 0
    weight_breakdown = {}
    
    for name, param in model.named_parameters():
        param_bytes = param.numel() * param.element_size()
        weight_memory_bytes += param_bytes
        weight_breakdown[name] = {
            'size': list(param.shape),
            'numel': param.numel(),
            'dtype': str(param.dtype),
            'memory_mb': param_bytes / (1024 ** 2)
        }
    
    weight_memory_mb = weight_memory_bytes / (1024 ** 2)
    
    # 2. Measure activation memory layer-wise (for breakdown only)
    # NOTE: This is theoretical - summing all activations as if they all exist simultaneously
    # In reality, activations are freed as forward pass progresses
    activation_memory_bytes_theoretical = 0
    activation_breakdown = {}
    
    def get_activation_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, dict):
                # Handle dict outputs (e.g., from some models)
                output = output.get('logits', list(output.values())[0])
            
            # Calculate activation memory
            if isinstance(output, torch.Tensor):
                act_bytes = output.numel() * output.element_size()
                activation_breakdown[layer_name] = {
                    'shape': list(output.shape),
                    'numel': output.numel(),
                    'dtype': str(output.dtype),
                    'memory_mb': act_bytes / (1024 ** 2)
                }
                nonlocal activation_memory_bytes_theoretical
                activation_memory_bytes_theoretical += act_bytes
        return hook
    
    # Register hooks on all major layers
    hooks = []
    
    # ResNet structure: conv1, bn1, maxpool, layer1-4, avgpool, fc
    if hasattr(model, 'conv1'):
        hooks.append(model.conv1.register_forward_hook(get_activation_hook('conv1')))
    if hasattr(model, 'bn1') and not isinstance(model.bn1, nn.Identity):
        hooks.append(model.bn1.register_forward_hook(get_activation_hook('bn1')))
    if hasattr(model, 'maxpool'):
        hooks.append(model.maxpool.register_forward_hook(get_activation_hook('maxpool')))
    
    # Register hooks on each block in layers
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for block_idx, block in enumerate(layer):
                # Hook on conv1 and conv2 of each block
                if hasattr(block, 'conv1'):
                    hooks.append(block.conv1.register_forward_hook(
                        get_activation_hook(f'{layer_name}.{block_idx}.conv1')))
                if hasattr(block, 'conv2'):
                    hooks.append(block.conv2.register_forward_hook(
                        get_activation_hook(f'{layer_name}.{block_idx}.conv2')))
                if hasattr(block, 'bn1') and not isinstance(block.bn1, nn.Identity):
                    hooks.append(block.bn1.register_forward_hook(
                        get_activation_hook(f'{layer_name}.{block_idx}.bn1')))
                if hasattr(block, 'bn2') and not isinstance(block.bn2, nn.Identity):
                    hooks.append(block.bn2.register_forward_hook(
                        get_activation_hook(f'{layer_name}.{block_idx}.bn2')))
                if hasattr(block, 'relu'):
                    hooks.append(block.relu.register_forward_hook(
                        get_activation_hook(f'{layer_name}.{block_idx}.relu')))
    
    if hasattr(model, 'avgpool'):
        hooks.append(model.avgpool.register_forward_hook(get_activation_hook('avgpool')))
    if hasattr(model, 'fc'):
        hooks.append(model.fc.register_forward_hook(get_activation_hook('fc')))
    
    # Run forward pass to capture activations
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    activation_memory_mb_theoretical = activation_memory_bytes_theoretical / (1024 ** 2)
    
    # 3. Measure peak memory (includes overhead, gradients, etc.)
    if device.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        # Actual activation memory = peak - weights - overhead
        # Overhead includes CUDA memory management, temporary buffers, etc.
        # Estimate overhead as 5% of peak (conservative estimate)
        estimated_overhead_mb = peak_memory_mb * 0.05
        actual_activation_memory_mb = max(0, peak_memory_mb - weight_memory_mb - estimated_overhead_mb)
        overhead_memory_mb = peak_memory_mb - weight_memory_mb - actual_activation_memory_mb
    else:
        peak_memory_mb = 0
        actual_activation_memory_mb = 0
        overhead_memory_mb = 0
    
    return {
        'weight_memory_mb': weight_memory_mb,
        'activation_memory_mb': actual_activation_memory_mb,  # Use actual, not theoretical
        'activation_memory_mb_theoretical': activation_memory_mb_theoretical,  # Keep for reference
        'overhead_memory_mb': overhead_memory_mb,
        'peak_memory_mb': peak_memory_mb,
        'weight_breakdown': weight_breakdown,
        'activation_breakdown': activation_breakdown,
        'is_fp16': is_fp16
    }


def print_memory_breakdown(memory_data):
    """Print detailed memory breakdown in a readable format."""
    print(f"\n{'='*80}")
    print(f"DETAILED MEMORY BREAKDOWN")
    print(f"{'='*80}")
    
    print(f"\n[WEIGHT MEMORY]")
    print(f"  Total Weight Memory: {memory_data['weight_memory_mb']:.2f} MB")
    if memory_data['is_fp16']:
        print(f"  Precision: FP16 (2 bytes per parameter)")
    else:
        print(f"  Precision: FP32 (4 bytes per parameter)")
    
    # Print top 10 largest weight layers
    weight_items = sorted(memory_data['weight_breakdown'].items(), 
                         key=lambda x: x[1]['memory_mb'], reverse=True)
    print(f"\n  Top 10 Largest Weight Layers:")
    for i, (name, info) in enumerate(weight_items[:10], 1):
        print(f"    {i:2d}. {name:40s}: {info['memory_mb']:8.3f} MB "
              f"({info['numel']:>10,} params, shape={info['size']})")
    
    print(f"\n[ACTIVATION MEMORY]")
    print(f"  Actual Peak Activation Memory: {memory_data['activation_memory_mb']:.2f} MB")
    if 'activation_memory_mb_theoretical' in memory_data:
        print(f"  (Theoretical sum if all activations existed simultaneously: {memory_data['activation_memory_mb_theoretical']:.2f} MB)")
    print(f"  Note: Activations are freed during forward pass, so actual peak is much lower than sum.")
    
    # Print activation breakdown by layer
    act_items = sorted(memory_data['activation_breakdown'].items(),
                      key=lambda x: x[1]['memory_mb'], reverse=True)
    print(f"\n  Layer-wise Activation Memory (theoretical - for reference only):")
    for name, info in act_items:
        print(f"    {name:40s}: {info['memory_mb']:8.3f} MB "
              f"(shape={info['shape']}, {info['numel']:>10,} elements)")
    
    print(f"\n[MEMORY SUMMARY]")
    if memory_data['peak_memory_mb'] > 0:
        weight_pct = (memory_data['weight_memory_mb'] / memory_data['peak_memory_mb'] * 100)
        act_pct = (memory_data['activation_memory_mb'] / memory_data['peak_memory_mb'] * 100)
        overhead_pct = (memory_data['overhead_memory_mb'] / memory_data['peak_memory_mb'] * 100)
        
        print(f"  Weight Memory:        {memory_data['weight_memory_mb']:8.2f} MB ({weight_pct:5.1f}%)")
        print(f"  Activation Memory:    {memory_data['activation_memory_mb']:8.2f} MB ({act_pct:5.1f}%)")
        print(f"  Overhead/Other:       {memory_data['overhead_memory_mb']:8.2f} MB ({overhead_pct:5.1f}%)")
        print(f"  {'-'*60}")
        print(f"  Peak GPU Memory:      {memory_data['peak_memory_mb']:8.2f} MB (100.0%)")
        
        print(f"\n[MEMORY EFFICIENCY]")
        print(f"  Weight/Peak Ratio:     {weight_pct:.1f}%")
        print(f"  Activation/Peak Ratio: {act_pct:.1f}%")
        print(f"  Overhead/Peak Ratio:   {overhead_pct:.1f}%")
    else:
        print(f"  Weight Memory:        {memory_data['weight_memory_mb']:8.2f} MB")
        print(f"  Activation Memory:    {memory_data['activation_memory_mb']:8.2f} MB")
        print(f"  Overhead/Other:       {memory_data['overhead_memory_mb']:8.2f} MB")
        print(f"  Peak GPU Memory:      {memory_data['peak_memory_mb']:8.2f} MB")


def measure_memory_and_latency(model, sample_input, device, num_iterations=100, warmup_iterations=10):
    """
    Measure GPU memory usage and inference latency with improved consistency.
    
    Uses CUDA events for accurate GPU timing and includes outlier removal
    for more consistent results.
    """
    model.eval()
    
    x = sample_input.to(device)
    
    # Check if model is FP16
    model_dtype = next(model.parameters()).dtype
    is_fp16 = model_dtype == torch.float16
    
    if is_fp16:
        x = x.half()
    
    print(f"\n[INFO] Measuring memory and latency...")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Measurement iterations: {num_iterations}")
    print(f"  Model dtype: {model_dtype}")
    
    # Clear memory before warmup
    clear_memory(device)
    
    # Extended warmup for better consistency
    print(f"  Warming up...")
    with torch.no_grad():
        for _ in range(warmup_iterations * 2):  # Double warmup for stability
            _ = model(x)
    
    # IMPORTANT: Wait after warmup for GPU to stabilize
    if device.type == 'cuda':
        torch.cuda.synchronize()
        time.sleep(0.5)  # Let GPU cool down and stabilize
    
    # Clear memory before measurement
    clear_memory(device)
    
    # Determine timing method (CUDA events are more accurate for GPU)
    use_cuda_events = False
    if device.type == 'cuda':
        try:
            # Test if CUDA events work
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            use_cuda_events = True
            print(f"  Using CUDA events for accurate GPU timing")
        except Exception as e:
            use_cuda_events = False
            print(f"  CUDA events not available, using time.perf_counter() instead")
    
    # Measure latency
    print(f"  Measuring latency...")
    latencies = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Ensure previous work is done
            
            if use_cuda_events:
                # CUDA events are more accurate for GPU timing
                start_event.record()
                _ = model(x)
                end_event.record()
                torch.cuda.synchronize()  # Wait for events to complete
                latency_ms = start_event.elapsed_time(end_event)  # Already in ms
            else:
                # Fallback to time.perf_counter() (more precise than time.time())
                start_time = time.perf_counter()
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0
            
            latencies.append(latency_ms)
            
            # Small delay every 10 iterations to prevent thermal buildup
            if (i + 1) % 10 == 0 and device.type == 'cuda':
                time.sleep(0.01)
    
    latencies = np.array(latencies)
    original_count = len(latencies)
    
    # Remove outliers (more than 3 standard deviations from mean) for consistency
    if len(latencies) > 10:
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        if std_lat > 0:
            mask = np.abs(latencies - mean_lat) < 3 * std_lat
            latencies_clean = latencies[mask]
            # Keep cleaned data if >80% of measurements remain
            if len(latencies_clean) > len(latencies) * 0.8:
                removed_count = len(latencies) - len(latencies_clean)
                latencies = latencies_clean
                if removed_count > 0:
                    print(f"  [INFO] Removed {removed_count} outlier measurements ({removed_count/original_count*100:.1f}%)")
    
    # Measure peak memory
    if device.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    else:
        peak_memory_mb = 0
    
    # Calculate statistics
    mean_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    p50_latency_ms = np.percentile(latencies, 50)
    p95_latency_ms = np.percentile(latencies, 95)
    p99_latency_ms = np.percentile(latencies, 99)
    min_latency_ms = np.min(latencies)
    max_latency_ms = np.max(latencies)
    
    # Coefficient of variation (CV) - measure of consistency
    cv = (std_latency_ms / mean_latency_ms * 100) if mean_latency_ms > 0 else 0
    
    batch_size = x.size(0)
    mean_sample_latency_ms = mean_latency_ms / batch_size
    throughput_samples_per_sec = 1000.0 / mean_sample_latency_ms if mean_sample_latency_ms > 0 else 0
    
    results = {
        'peak_memory_mb': peak_memory_mb,
        'mean_latency_ms': mean_latency_ms,
        'std_latency_ms': std_latency_ms,
        'p50_latency_ms': p50_latency_ms,
        'p95_latency_ms': p95_latency_ms,
        'p99_latency_ms': p99_latency_ms,
        'min_latency_ms': min_latency_ms,
        'max_latency_ms': max_latency_ms,
        'mean_sample_latency_ms': mean_sample_latency_ms,
        'throughput_samples_per_sec': throughput_samples_per_sec,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'is_fp16': is_fp16,
        'cv_percent': cv,  # Coefficient of variation (lower = more consistent)
        'valid_iterations': len(latencies)  # Number of valid measurements after outlier removal
    }
    
    return results


def load_cifar10_test(data_dir="./data", batch_size=128, normalization=None):
    """
    Load CIFAR-10 test dataset. Automatically downloads if not present.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoader
        normalization: Optional dict with 'mean' and 'std' keys. If None, uses default CIFAR-10 stats.
    """
    # Use provided normalization or default CIFAR-10 normalization
    if normalization is None:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    else:
        normalize = transforms.Normalize(
            mean=normalization['mean'],
            std=normalization['std']
        )
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Always download if not present
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    return test_loader


@torch.no_grad()
def evaluate_accuracy(model, test_loader, device, max_samples=None):
    """
    Evaluate model accuracy on test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        accuracy: Test accuracy percentage
    """
    model.eval()
    
    correct = 0
    total = 0
    sample_count = 0
    
    print(f"\n[INFO] Evaluating model on test dataset...")
    if max_samples:
        print(f"  (Limiting to {max_samples} samples)")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if max_samples and sample_count >= max_samples:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            # Ensure input dtype matches model dtype (pipeline uses FP16, we use FP32)
            model_dtype = next(model.parameters()).dtype
            if model_dtype == torch.float16:
                x = x.half()  # Match pipeline: convert input to FP16 if model is FP16
            # Otherwise keep FP32 (default)
            
            # Handle model forward pass (match pipeline's evaluate_quick exactly)
            # Pipeline tries: model(x, training=False, return_masks=False, return_saliency=False)
            # Then falls back to: model(x)
            # Also handle CUDA errors by converting to FP32 if needed
            try:
                output = model(x, training=False, return_masks=False, return_saliency=False)
            except TypeError:
                try:
                    output = model(x, training=False)
                except TypeError:
                    try:
                        output = model(x)
                    except RuntimeError as e:
                        if 'CUBLAS_STATUS_NOT_SUPPORTED' in str(e) or 'cublas' in str(e).lower():
                            # Convert model and input to FP32
                            model = model.float()
                            x = x.float()
                            output = model(x)
                        else:
                            raise
            except RuntimeError as e:
                if 'CUBLAS_STATUS_NOT_SUPPORTED' in str(e) or 'cublas' in str(e).lower():
                    # Convert model and input to FP32
                    model = model.float()
                    x = x.float()
                    output = model(x)
                else:
                    raise
            
            # Get logits if output is tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # DEBUG: Print first batch info
            if batch_idx == 0:
                print(f"  DEBUG - First batch:")
                print(f"    Input shape: {x.shape}")
                print(f"    Output shape: {logits.shape}")
                print(f"    Output min/max/mean: {logits.min().item():.4f} / {logits.max().item():.4f} / {logits.mean().item():.4f}")
                print(f"    Output std: {logits.std().item():.4f}")
                print(f"    Predictions: {logits.argmax(dim=1)[:5].cpu().numpy()}")
                print(f"    True labels: {y[:5].cpu().numpy()}")
            
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            sample_count += y.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * correct / total
                print(f"  Batch {batch_idx + 1}: {current_acc:.2f}% ({correct}/{total})")
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"  Final Accuracy: {accuracy:.2f}% ({correct}/{total} samples)")
    
    return accuracy


def print_results(results, checkpoint_data, measured_accuracy=None):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*80}")
    
    print(f"\n[MEMORY METRICS]")
    print(f"  Peak GPU Memory: {results['peak_memory_mb']:.2f} MB")
    if results['is_fp16']:
        print(f"  Precision: FP16 (50% memory reduction vs FP32)")
    
    print(f"\n[LATENCY METRICS (Batch)]")
    print(f"  Mean:   {results['mean_latency_ms']:.3f} ± {results['std_latency_ms']:.3f} ms")
    print(f"  P50:    {results['p50_latency_ms']:.3f} ms")
    print(f"  P95:    {results['p95_latency_ms']:.3f} ms")
    print(f"  P99:    {results['p99_latency_ms']:.3f} ms")
    print(f"  Min:    {results['min_latency_ms']:.3f} ms")
    print(f"  Max:    {results['max_latency_ms']:.3f} ms")
    
    # Show consistency metric (Coefficient of Variation)
    cv = results.get('cv_percent', 0)
    consistency_status = "Excellent" if cv < 2.0 else "Good" if cv < 5.0 else "Fair" if cv < 10.0 else "Variable"
    print(f"  CV:     {cv:.2f}% ({consistency_status} consistency)")
    
    print(f"\n[LATENCY METRICS (Per-Sample)]")
    print(f"  Mean:   {results['mean_sample_latency_ms']:.3f} ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
    
    print(f"\n[MODEL INFO]")
    if measured_accuracy is not None:
        print(f"  Test Accuracy (measured): {measured_accuracy:.2f}%")
        checkpoint_acc = checkpoint_data.get('final_test_acc', 0)
        if checkpoint_acc > 0:
            print(f"  Test Accuracy (from checkpoint): {checkpoint_acc:.2f}%")
            diff = measured_accuracy - checkpoint_acc
            print(f"  Difference: {diff:+.2f}%")
    else:
        checkpoint_acc = checkpoint_data.get('final_test_acc', 0)
        if checkpoint_acc > 0:
            print(f"  Test Accuracy (from checkpoint): {checkpoint_acc:.2f}%")
        else:
            print(f"  Test Accuracy: Not measured (use --evaluate to measure)")
    print(f"  Number of Classes: {checkpoint_data.get('num_classes', 10)}")
    print(f"  Channel Counts: {checkpoint_data.get('channel_counts', {})}")
    print(f"  Batch Size: {results['batch_size']}")
    print(f"  Iterations: {results.get('valid_iterations', results['num_iterations'])}/{results['num_iterations']} (valid/total)")


def plot_inference_comparison(model, checkpoint_data, device, batch_size=32, input_size=32, save_path=None):
    """
    Generate inference comparison plots.
    Compares pruned model against standard ResNet-18 baseline (both FP32 and FP16).
    
    Args:
        model: The pruned model to compare
        checkpoint_data: Checkpoint metadata
        device: Device to run on
        batch_size: Batch size to use for both models (default: 32)
        input_size: Input image size (default: 32)
        save_path: Optional path to save plot
    """
    try:
        print(f"\n[INFO] Generating inference comparison plots...")
        print(f"  Using batch_size={batch_size}, input_size={input_size}x{input_size} for all models")
        
        # Determine dataset from checkpoint
        num_classes = checkpoint_data.get('num_classes', 10)
        dataset_name = 'CIFAR-10' if num_classes == 10 else f'ImageNet-{num_classes}' if num_classes == 100 else f'{num_classes} classes'
        
        # Create standard ResNet-18 baseline with CIFAR-10 configuration (stride=1 for conv1/maxpool)
        # This matches the pruned model's configuration for fair comparison
        standard_model_fp32 = resnet18(num_classes=num_classes).to(device)
        # Modify for CIFAR-10: conv1 stride=1, maxpool stride=1 (matches gated/pruned model)
        standard_model_fp32.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
        standard_model_fp32.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to(device)
        standard_model_fp32.eval()
        # Keep FP32 for baseline comparison
        
        # Create FP16 version of baseline
        standard_model_fp16 = copy.deepcopy(standard_model_fp32)
        standard_model_fp16 = standard_model_fp16.half()
        
        # Create dummy inputs
        dummy_input_fp32 = torch.randn(batch_size, 3, input_size, input_size, dtype=torch.float32).to(device)
        dummy_input_fp16 = torch.randn(batch_size, 3, input_size, input_size, dtype=torch.float16).to(device)
        
        # Clear memory before measurements
        clear_memory(device)
        
        # Measure FP32 baseline model
        print("  Measuring standard ResNet-18 (FP32)...")
        standard_results_fp32 = measure_memory_and_latency(
            standard_model_fp32, dummy_input_fp32, device, num_iterations=50, warmup_iterations=5
        )
        
        print(f"\n[BASELINE MODEL - Standard ResNet-18 FP32]")
        print(f"  Peak GPU Memory: {standard_results_fp32['peak_memory_mb']:.2f} MB")
        print(f"  Precision: FP32 (4 bytes per parameter)")
        print(f"  Batch Latency: {standard_results_fp32['mean_latency_ms']:.3f} ± {standard_results_fp32['std_latency_ms']:.3f} ms")
        print(f"  Per-Sample Latency: {standard_results_fp32['mean_sample_latency_ms']:.3f} ms")
        print(f"  Throughput: {standard_results_fp32['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Move FP32 model to CPU
        standard_model_fp32 = standard_model_fp32.cpu()
        clear_memory(device)
        
        # Measure FP16 baseline model
        print("\n  Measuring standard ResNet-18 (FP16)...")
        standard_results_fp16 = measure_memory_and_latency(
            standard_model_fp16, dummy_input_fp16, device, num_iterations=50, warmup_iterations=5
        )
        
        print(f"\n[BASELINE MODEL - Standard ResNet-18 FP16]")
        print(f"  Peak GPU Memory: {standard_results_fp16['peak_memory_mb']:.2f} MB")
        print(f"  Precision: FP16 (2 bytes per parameter)")
        print(f"  Batch Latency: {standard_results_fp16['mean_latency_ms']:.3f} ± {standard_results_fp16['std_latency_ms']:.3f} ms")
        print(f"  Per-Sample Latency: {standard_results_fp16['mean_sample_latency_ms']:.3f} ms")
        print(f"  Throughput: {standard_results_fp16['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Move FP16 baseline to CPU temporarily
        standard_model_fp16 = standard_model_fp16.cpu()
        clear_memory(device)
        
        # Measure pruned model
        print("\n  Measuring pruned model...")
        pruned_results = measure_memory_and_latency(
            model, dummy_input_fp16, device, num_iterations=50, warmup_iterations=5
        )
        
        # Print pruned model results for comparison
        print(f"\n[PRUNED MODEL]")
        print(f"  Peak GPU Memory: {pruned_results['peak_memory_mb']:.2f} MB")
        print(f"  Precision: {'FP16' if pruned_results['is_fp16'] else 'FP32'} ({2 if pruned_results['is_fp16'] else 4} bytes per parameter)")
        print(f"  Batch Latency: {pruned_results['mean_latency_ms']:.3f} ± {pruned_results['std_latency_ms']:.3f} ms")
        print(f"  Per-Sample Latency: {pruned_results['mean_sample_latency_ms']:.3f} ms")
        print(f"  Throughput: {pruned_results['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Keep a reference to FP16 standard model for layer-wise comparison
        standard_model_for_comparison = standard_model_fp16.to(device)
        standard_model_for_comparison.eval()
        
        # Generate layer-wise comparison plots (using FP16 baseline)
        layerwise_path = save_path.replace('.png', '_layerwise.png') if save_path and save_path.endswith('.png') else (save_path + '_layerwise' if save_path else None)
        layerwise_result = plot_layerwise_memory_comparison(
            model, standard_model_for_comparison, checkpoint_data, device,
            batch_size=batch_size, input_size=input_size, save_path=layerwise_path
        )
        if layerwise_result is None:
            print("  ⚠️  Warning: Layer-wise comparison plots generation failed. Check error messages above.")
        
        # Move standard model back to CPU
        standard_model_for_comparison = standard_model_for_comparison.cpu()
        clear_memory(device)
        
        # Get GPU specifications with fallback methods
        gpu_name = "CPU"
        cuda_version = "N/A"
        gpu_memory_gb = "N/A"
        driver_version = "N/A"
        compute_cap = "N/A"
        
        if device.type == 'cuda':
            try:
                # Method 1: Try nvidia-smi for detailed info
                import subprocess
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', 
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True, timeout=5
                    )
                    parts = [p.strip() for p in result.stdout.strip().split(',')]
                    if len(parts) >= 4:
                        gpu_name = parts[0]
                        driver_version = parts[1]
                        gpu_memory_gb = f"{float(parts[2]) / 1024:.1f} GB"  # Convert MB to GB
                        compute_cap = parts[3]
                        cuda_version = torch.version.cuda  # Still get CUDA version from PyTorch
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
                    # Method 2: Fallback to PyTorch
                    gpu_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda
                    gpu_memory_gb = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                    compute_cap = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            except Exception as e:
                print(f"[WARN] Could not retrieve GPU info: {e}")
        
        # Get data size information
        # Try to get actual test dataset size from checkpoint or estimate
        num_test_samples = checkpoint_data.get('num_test_samples', None)
        if num_test_samples is None:
            # Estimate based on common datasets
            if num_classes == 10:
                num_test_samples = 10000  # CIFAR-10 test set
            elif num_classes == 100:
                num_test_samples = 5000   # ImageNet-100 test set (typical split)
            else:
                num_test_samples = 10000   # Default estimate
        
        # Create comparison plot with 3 models
        fig = plt.figure(figsize=(20, 12.0))
        gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.3, top=0.88, bottom=0.08)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        
        # Main title
        fig.suptitle('NeuroLattice Inference Comparison',
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Subtitle with dataset, batch size, input size, and samples
        fig.text(0.5, 0.92, f'{dataset_name} | Batch Size: {batch_size} | Input Size: {input_size}×{input_size} | Samples: {num_test_samples:,}',
                ha='center', fontsize=12, style='italic')
        
        models = ['Standard\nResNet-18\n(FP32)', 'Standard\nResNet-18\n(FP16)', 'NeuroLattice\nResNet-18\n(FP16)']
        colors = ['#888888', '#555555', 'green']  # Light gray, dark gray, green
        
        # Plot 1: Memory comparison
        ax = axes[0]
        memory_mb = [standard_results_fp32['peak_memory_mb'], 
                     standard_results_fp16['peak_memory_mb'], 
                     pruned_results['peak_memory_mb']]
        bars = ax.bar(models, memory_mb, color=colors, alpha=0.7)
        ax.set_ylabel('Peak GPU Memory (MB)')
        ax.set_title('GPU Memory Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, mem in zip(bars, memory_mb):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem:.2f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Calculate reductions vs FP32 and FP16 baselines
        reduction_vs_fp32 = (1 - pruned_results['peak_memory_mb'] / standard_results_fp32['peak_memory_mb']) * 100
        reduction_vs_fp16 = (1 - pruned_results['peak_memory_mb'] / standard_results_fp16['peak_memory_mb']) * 100
        ax.text(0.5, max(memory_mb) * 0.65,
               f'vs FP32: {reduction_vs_fp32:.1f}% reduction\nvs FP16: {reduction_vs_fp16:.1f}% reduction',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Latency comparison
        ax = axes[1]
        time_per_sample = [standard_results_fp32['mean_sample_latency_ms'], 
                          standard_results_fp16['mean_sample_latency_ms'], 
                          pruned_results['mean_sample_latency_ms']]
        bars = ax.bar(models, time_per_sample, color=colors, alpha=0.7)
        ax.set_ylabel('Time per Sample (ms)')
        ax.set_title('Inference Speed Comparison\n(Lower is Better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, t in zip(bars, time_per_sample):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.3f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        speedup_vs_fp32 = standard_results_fp32['mean_sample_latency_ms'] / pruned_results['mean_sample_latency_ms']
        speedup_vs_fp16 = standard_results_fp16['mean_sample_latency_ms'] / pruned_results['mean_sample_latency_ms']
        ax.text(0.5, max(time_per_sample) * 0.65,
               f'vs FP32: {speedup_vs_fp32:.2f}x speedup\nvs FP16: {speedup_vs_fp16:.2f}x speedup',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Plot 3: Throughput comparison
        ax = axes[2]
        throughputs = [standard_results_fp32['throughput_samples_per_sec'], 
                      standard_results_fp16['throughput_samples_per_sec'], 
                      pruned_results['throughput_samples_per_sec']]
        bars = ax.bar(models, throughputs, color=colors, alpha=0.7)
        ax.set_ylabel('Throughput (Samples/sec)')
        ax.set_title('Throughput Comparison\n(Higher is Better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, tp in zip(bars, throughputs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{tp:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        improvement_vs_fp32 = (pruned_results['throughput_samples_per_sec'] / standard_results_fp32['throughput_samples_per_sec'] - 1) * 100
        improvement_vs_fp16 = (pruned_results['throughput_samples_per_sec'] / standard_results_fp16['throughput_samples_per_sec'] - 1) * 100
        ax.text(0.5, max(throughputs) * 0.65,
               f'vs FP32: {improvement_vs_fp32:+.1f}% improvement\nvs FP16: {improvement_vs_fp16:+.1f}% improvement',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Add GPU specifications below the plots
        gpu_specs_ax = fig.add_subplot(gs[1, :])
        gpu_specs_ax.axis('off')
        # Build comprehensive GPU specs string
        gpu_specs_parts = [f'GPU: {gpu_name}', f'CUDA: {cuda_version}']
        if driver_version != "N/A":
            gpu_specs_parts.append(f'Driver: {driver_version}')
        gpu_specs_parts.append(f'Memory: {gpu_memory_gb}')
        if compute_cap != "N/A":
            gpu_specs_parts.append(f'Compute Capability: {compute_cap}')
        gpu_specs_text = ' | '.join(gpu_specs_parts)
        gpu_specs_ax.text(0.5, 0.5, gpu_specs_text, 
                        ha='center', va='center', fontsize=11, 
                        color='black',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5),
                        family='monospace', fontweight='normal')
        
        plt.tight_layout()
        
        # Determine save path
        if save_path is None:
            save_path = 'inference_comparison.png'
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot to: {os.path.abspath(save_path)}")
        
        plt.close()
        
        return {
            'standard_fp32': standard_results_fp32,
            'standard_fp16': standard_results_fp16,
            'pruned': pruned_results,
            'memory_reduction_vs_fp32_pct': reduction_vs_fp32,
            'memory_reduction_vs_fp16_pct': reduction_vs_fp16,
            'speedup_vs_fp32': speedup_vs_fp32,
            'speedup_vs_fp16': speedup_vs_fp16,
            'throughput_improvement_vs_fp32_pct': improvement_vs_fp32,
            'throughput_improvement_vs_fp16_pct': improvement_vs_fp16
        }
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_layerwise_comparison(baseline_weights, pruned_weights, baseline_activations, pruned_activations,
                               layer_order, activation_layer_order):
    """
    Print layer-wise weight and activation memory comparison in a readable table format.
    """
    print(f"\n{'='*100}")
    print(f"LAYER-WISE MEMORY COMPARISON (Baseline vs Pruned)")
    print(f"{'='*100}")
    
    # Print weight memory comparison
    print(f"\n{'WEIGHT MEMORY COMPARISON':^100}")
    print(f"{'-'*100}")
    print(f"{'Layer':<40} {'Baseline (MB)':>15} {'Pruned (MB)':>15} {'Reduction %':>15} {'Reduction':>15}")
    print(f"{'-'*100}")
    
    total_baseline_weights = 0
    total_pruned_weights = 0
    
    for name in layer_order:
        baseline_val = baseline_weights.get(name, 0)
        pruned_val = pruned_weights.get(name, 0)
        if baseline_val > 0 or pruned_val > 0:
            total_baseline_weights += baseline_val
            total_pruned_weights += pruned_val
            reduction_pct = ((baseline_val - pruned_val) / baseline_val * 100) if baseline_val > 0 else 0
            reduction_mb = baseline_val - pruned_val
            print(f"{name:<40} {baseline_val:>15.4f} {pruned_val:>15.4f} {reduction_pct:>14.2f}% {reduction_mb:>14.4f} MB")
    
    print(f"{'-'*100}")
    total_reduction_pct = ((total_baseline_weights - total_pruned_weights) / total_baseline_weights * 100) if total_baseline_weights > 0 else 0
    total_reduction_mb = total_baseline_weights - total_pruned_weights
    print(f"{'TOTAL':<40} {total_baseline_weights:>15.4f} {total_pruned_weights:>15.4f} {total_reduction_pct:>14.2f}% {total_reduction_mb:>14.4f} MB")
    
    # Print activation memory comparison
    print(f"\n{'ACTIVATION MEMORY COMPARISON (Theoretical)':^100}")
    print(f"{'-'*100}")
    print(f"{'Layer':<40} {'Baseline (MB)':>15} {'Pruned (MB)':>15} {'Reduction %':>15} {'Reduction':>15}")
    print(f"{'-'*100}")
    
    total_baseline_acts = 0
    total_pruned_acts = 0
    
    for layer_name in activation_layer_order:
        baseline_act_data = baseline_activations.get(layer_name, {})
        pruned_act_data = pruned_activations.get(layer_name, {})
        baseline_val = baseline_act_data.get('memory_mb', 0.0) if isinstance(baseline_act_data, dict) else 0.0
        pruned_val = pruned_act_data.get('memory_mb', 0.0) if isinstance(pruned_act_data, dict) else 0.0
        
        if baseline_val > 0 or pruned_val > 0:
            total_baseline_acts += baseline_val
            total_pruned_acts += pruned_val
            reduction_pct = ((baseline_val - pruned_val) / baseline_val * 100) if baseline_val > 0 else 0
            reduction_mb = baseline_val - pruned_val
            print(f"{layer_name:<40} {baseline_val:>15.4f} {pruned_val:>15.4f} {reduction_pct:>14.2f}% {reduction_mb:>14.4f} MB")
    
    print(f"{'-'*100}")
    total_act_reduction_pct = ((total_baseline_acts - total_pruned_acts) / total_baseline_acts * 100) if total_baseline_acts > 0 else 0
    total_act_reduction_mb = total_baseline_acts - total_pruned_acts
    print(f"{'TOTAL':<40} {total_baseline_acts:>15.4f} {total_pruned_acts:>15.4f} {total_act_reduction_pct:>14.2f}% {total_act_reduction_mb:>14.4f} MB")
    print(f"{'='*100}\n")


def plot_layerwise_memory_comparison(pruned_model, baseline_model, checkpoint_data, device, 
                                     batch_size=32, input_size=32, save_path=None):
    """
    Generate layer-wise weight and activation memory comparison plots.
    Compares pruned model against standard ResNet-18 baseline layer by layer.
    
    Args:
        pruned_model: The pruned model to compare
        baseline_model: The baseline ResNet-18 model (FP16)
        checkpoint_data: Checkpoint metadata
        device: Device to run on
        batch_size: Batch size to use for both models
        input_size: Input image size
        save_path: Optional path to save plot (without extension, will add _weights.png and _activations.png)
    """
    try:
        print(f"\n[INFO] Generating layer-wise memory comparison plots...")
        print(f"  Using batch_size={batch_size}, input_size={input_size}x{input_size} for both models")
        print(f"  Note: If you see OpenMP errors, the plots may fail. Comparison table will still be printed.")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size, input_size, dtype=torch.float16).to(device)
        
        # Measure detailed memory breakdown for baseline
        print("  Measuring baseline model layer-wise memory...")
        clear_memory(device)
        baseline_memory = measure_detailed_memory_breakdown(baseline_model, dummy_input, device)
        
        # Print baseline model memory breakdown
        print(f"\n{'='*100}")
        print(f"BASELINE MODEL (Standard ResNet-18 FP16) - LAYER-WISE MEMORY BREAKDOWN")
        print(f"{'='*100}")
        print_memory_breakdown(baseline_memory)
        
        # Move baseline to CPU and clear memory
        baseline_model = baseline_model.cpu()
        clear_memory(device)
        
        # Measure detailed memory breakdown for pruned
        print("  Measuring pruned model layer-wise memory...")
        pruned_memory = measure_detailed_memory_breakdown(pruned_model, dummy_input, device)
        
        # Normalize layer names for comparison (remove .weight suffix, handle conv/bn naming)
        def normalize_layer_name(name):
            """Normalize layer names for comparison."""
            # Remove .weight suffix
            name = name.replace('.weight', '')
            name = name.replace('.bias', '')  # Also handle bias
            # Handle conv layers - use conv name
            if '.conv' in name:
                # For PrunedConv2d, the weight is at .conv.weight
                name = name.replace('.conv', '')
            return name
        
        def find_matching_weight(param_name, target_name):
            """Find if param_name matches target_name (flexible matching)."""
            norm_param = normalize_layer_name(param_name)
            # Direct match
            if norm_param == target_name:
                return True
            # Check if target_name is contained in norm_param (e.g., "layer1.0.conv1" in "layer1.0.conv1.weight")
            if target_name in norm_param:
                return True
            # Check if norm_param is contained in target_name
            if norm_param in target_name:
                return True
            return False
        
        # Create layer name mapping for consistent ordering
        layer_order = [
            'conv1', 'bn1', 'maxpool',
            'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2',
            'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2',
            'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2',
            'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2',
            'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2',
            'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2',
            'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.conv2', 'layer4.0.bn2',
            'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.conv2', 'layer4.1.bn2',
            'avgpool', 'fc'
        ]
        
        # Collect weight memory by layer
        baseline_weights = {}
        pruned_weights = {}
        
        # Debug: print sample weight breakdown keys
        print(f"    Sample baseline weight keys: {list(baseline_memory['weight_breakdown'].keys())[:10]}")
        print(f"    Sample pruned weight keys: {list(pruned_memory['weight_breakdown'].keys())[:10]}")
        
        for name in layer_order:
            # Skip bn layers if BatchNorm is fused (they don't exist as separate parameters)
            if name.endswith('.bn1') or name.endswith('.bn2') or name == 'bn1':
                # Check if BatchNorm is fused by looking for biases in conv layers
                if baseline_memory.get('is_fp16', False) or pruned_memory.get('is_fp16', False):
                    # In fused models, bn layers don't have separate weights
                    baseline_weights[name] = 0.0
                    pruned_weights[name] = 0.0
                    continue
            
            # Try to find matching weight in baseline
            baseline_found = False
            baseline_mem = 0.0
            for param_name, param_data in baseline_memory['weight_breakdown'].items():
                if find_matching_weight(param_name, name):
                    baseline_mem += param_data['memory_mb']  # Sum in case multiple matches (weight + bias)
                    baseline_found = True
            baseline_weights[name] = baseline_mem if baseline_found else 0.0
            
            # Try to find matching weight in pruned
            pruned_found = False
            pruned_mem = 0.0
            for param_name, param_data in pruned_memory['weight_breakdown'].items():
                if find_matching_weight(param_name, name):
                    pruned_mem += param_data['memory_mb']  # Sum in case multiple matches (weight + bias)
                    pruned_found = True
            pruned_weights[name] = pruned_mem if pruned_found else 0.0
        
        # Collect activation memory by layer (use theoretical for layer-wise comparison)
        baseline_activations = baseline_memory.get('activation_breakdown', {})
        pruned_activations = pruned_memory.get('activation_breakdown', {})
        
        print(f"    Baseline activation breakdown keys: {len(baseline_activations)}")
        print(f"    Pruned activation breakdown keys: {len(pruned_activations)}")
        if len(baseline_activations) > 0:
            print(f"    Sample baseline keys: {list(baseline_activations.keys())[:5]}")
        if len(pruned_activations) > 0:
            print(f"    Sample pruned keys: {list(pruned_activations.keys())[:5]}")
        
        # Create activation layer mapping
        activation_layer_order = [
            'conv1', 'bn1', 'maxpool',
            'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu',
            'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu',
            'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu',
            'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu',
            'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu',
            'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu',
            'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu',
            'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu',
            'avgpool', 'fc'
        ]
        
        # Filter to only layers that exist in at least one model
        baseline_act_values = []
        pruned_act_values = []
        act_layer_labels = []
        
        for layer_name in activation_layer_order:
            baseline_act_data = baseline_activations.get(layer_name, {})
            pruned_act_data = pruned_activations.get(layer_name, {})
            baseline_val = baseline_act_data.get('memory_mb', 0.0) if isinstance(baseline_act_data, dict) else 0.0
            pruned_val = pruned_act_data.get('memory_mb', 0.0) if isinstance(pruned_act_data, dict) else 0.0
            if baseline_val > 0 or pruned_val > 0:
                baseline_act_values.append(baseline_val)
                pruned_act_values.append(pruned_val)
                act_layer_labels.append(layer_name.replace('.', '\n'))
        
        # Initialize plot paths
        weight_save_path = None
        act_save_path = None
        
        # Create weight comparison plot
        print(f"  Creating weight comparison plot...")
        baseline_nonzero = [v for v in baseline_weights.values() if v > 0]
        pruned_nonzero = [v for v in pruned_weights.values() if v > 0]
        print(f"    Baseline weight layers found: {len(baseline_nonzero)}")
        print(f"    Pruned weight layers found: {len(pruned_nonzero)}")
        
        # Filter layers that have data in at least one model
        weight_layer_labels = []
        baseline_weight_values = []
        pruned_weight_values = []
        
        for name in layer_order:
            baseline_val = baseline_weights.get(name, 0)
            pruned_val = pruned_weights.get(name, 0)
            if baseline_val > 0 or pruned_val > 0:
                weight_layer_labels.append(name.replace('.', '\n'))
                baseline_weight_values.append(baseline_val)
                pruned_weight_values.append(pruned_val)
        
        if len(weight_layer_labels) == 0:
            print("  ⚠️  Warning: No weight layers found for comparison!")
            weight_save_path = None
        else:
            print(f"    Plotting {len(weight_layer_labels)} weight layers...")
            print(f"    Attempting to create matplotlib figure...")
            weight_save_path = None
            fig1 = None
            try:
                import sys
                sys.stdout.flush()  # Ensure output is flushed before matplotlib call
                fig1, ax1 = plt.subplots(figsize=(16, 8))
                print(f"    ✅ Figure created successfully")
                
                x = np.arange(len(weight_layer_labels))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, baseline_weight_values, width, label='Baseline (FP16)', color='#555555', alpha=0.7)
                bars2 = ax1.bar(x + width/2, pruned_weight_values, width, label='Pruned (FP16)', color='green', alpha=0.7)
                
                ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Weight Memory (MB)', fontsize=12, fontweight='bold')
                ax1.set_title('Layer-wise Weight Memory Comparison\n(Baseline vs Pruned ResNet-18, FP16)', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(weight_layer_labels, rotation=45, ha='right', fontsize=8)
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.001:  # Only show label if significant
                            ax1.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
                
                plt.tight_layout()
                
                # Save weight plot
                if save_path is None:
                    weight_save_path = 'layerwise_weights_comparison.png'
                else:
                    weight_save_path = save_path.replace('.png', '_weights.png') if save_path.endswith('.png') else f'{save_path}_weights.png'
                
                save_dir = os.path.dirname(weight_save_path) if os.path.dirname(weight_save_path) else '.'
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                plt.savefig(weight_save_path, dpi=150, bbox_inches='tight')
                print(f"✅ Saved weight comparison plot to: {os.path.abspath(weight_save_path)}")
            except Exception as e:
                print(f"❌ Error creating/saving weight plot: {e}")
                import traceback
                traceback.print_exc()
                weight_save_path = None
            finally:
                if fig1 is not None:
                    try:
                        plt.close(fig1)
                    except:
                        pass
        
        # Create activation comparison plot
        print(f"  Creating activation comparison plot...")
        print(f"    Baseline activation layers found: {len(baseline_act_values)}")
        print(f"    Pruned activation layers found: {len(pruned_act_values)}")
        
        if len(act_layer_labels) == 0:
            print("  ⚠️  Warning: No activation layers found for comparison!")
            act_save_path = None
        else:
            act_save_path = None
            fig2 = None
            try:
                fig2, ax2 = plt.subplots(figsize=(16, 8))
                
                x = np.arange(len(act_layer_labels))
                width = 0.35
            
                bars1 = ax2.bar(x - width/2, baseline_act_values, width, label='Baseline (FP16)', color='#555555', alpha=0.7)
                bars2 = ax2.bar(x + width/2, pruned_act_values, width, label='Pruned (FP16)', color='green', alpha=0.7)
                
                ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Activation Memory (MB)', fontsize=12, fontweight='bold')
                ax2.set_title('Layer-wise Activation Memory Comparison\n(Baseline vs Pruned ResNet-18, FP16, Theoretical)', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(act_layer_labels, rotation=45, ha='right', fontsize=8)
                ax2.legend(fontsize=11)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars (only for significant values)
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.01:  # Only show label if significant
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
                
                plt.tight_layout()
                
                # Save activation plot
                if save_path is None:
                    act_save_path = 'layerwise_activations_comparison.png'
                else:
                    act_save_path = save_path.replace('.png', '_activations.png') if save_path.endswith('.png') else f'{save_path}_activations.png'
                
                save_dir = os.path.dirname(act_save_path) if os.path.dirname(act_save_path) else '.'
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                plt.savefig(act_save_path, dpi=150, bbox_inches='tight')
                print(f"✅ Saved activation comparison plot to: {os.path.abspath(act_save_path)}")
            except Exception as e:
                print(f"❌ Error creating/saving activation plot: {e}")
                import traceback
                traceback.print_exc()
                act_save_path = None
            finally:
                if fig2 is not None:
                    try:
                        plt.close(fig2)
                    except:
                        pass
        
        # Print layer-wise comparison to console (do this even if plotting fails)
        print(f"\n{'='*100}")
        print(f"PRINTING LAYER-WISE MEMORY COMPARISON TABLE...")
        print(f"{'='*100}")
        try:
            print_layerwise_comparison(baseline_weights, pruned_weights, baseline_activations, pruned_activations, 
                                       layer_order, activation_layer_order)
        except Exception as e:
            print(f"⚠️  Warning: Could not print layer-wise comparison: {e}")
            import traceback
            traceback.print_exc()
        
        result = {
            'baseline_memory': baseline_memory,
            'pruned_memory': pruned_memory,
        }
        if 'weight_save_path' in locals():
            result['weight_save_path'] = weight_save_path
        if 'act_save_path' in locals() and act_save_path is not None:
            result['activation_save_path'] = act_save_path
        
        print(f"✅ Layer-wise comparison completed!")
        return result
    except Exception as e:
        print(f"\n❌ Error generating layer-wise plots: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='ResNet Model Inference with Memory and Latency Measurement')
    parser.add_argument('--checkpoint', type=str, 
                       default='model.pt',
                       help='Path to model checkpoint file (.pt)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of iterations for latency measurement (default: 100)')
    parser.add_argument('--warmup-iterations', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--input-size', type=int, default=32,
                       help='Input image size (default: 32 for CIFAR-10)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--plot-path', type=str, default=None,
                       help='Path to save plot (default: inference_comparison.png)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Measure accuracy on CIFAR-10 test set (auto-downloads if not present)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--max-eval-samples', type=int, default=None,
                       help='Maximum number of test samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    try:
        model, checkpoint_data = load_model_from_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n[INFO] Creating dummy input (batch_size={args.batch_size}, size={args.input_size}x{args.input_size})...")
    dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
    
    # Measure detailed memory breakdown
    print(f"\n[INFO] Measuring detailed memory breakdown...")
    memory_breakdown = measure_detailed_memory_breakdown(model, dummy_input, device)
    print_memory_breakdown(memory_breakdown)
    
    results = measure_memory_and_latency(
        model, 
        dummy_input, 
        device,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations
    )
    
    # Measure accuracy if requested
    measured_accuracy = None
    if args.evaluate:
        try:
            print(f"\n[INFO] Loading CIFAR-10 test dataset from: {args.data_dir}")
            print("  (Will auto-download if not present)")
            
            # Use data_preprocessing from checkpoint if available
            normalization = None
            if checkpoint_data and checkpoint_data.get('data_preprocessing'):
                data_prep = checkpoint_data['data_preprocessing']
                normalization = data_prep.get('normalization')
                print(f"[INFO] Using normalization from checkpoint: mean={normalization.get('mean')}, std={normalization.get('std')}")
            else:
                print(f"[INFO] Using default CIFAR-10 normalization (checkpoint doesn't have data_preprocessing)")
            
            test_loader = load_cifar10_test(
                data_dir=args.data_dir, 
                batch_size=args.batch_size,
                normalization=normalization
            )
            measured_accuracy = evaluate_accuracy(
                model, 
                test_loader, 
                device,
                max_samples=args.max_eval_samples
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Could not evaluate accuracy: {e}")
            print("  Continuing without accuracy measurement...")
            import traceback
            traceback.print_exc()
    
    print_results(results, checkpoint_data, measured_accuracy=measured_accuracy)
    
    if args.plot:
        try:
            plot_results = plot_inference_comparison(
                model, 
                checkpoint_data, 
                device,
                batch_size=args.batch_size,
                input_size=args.input_size,
                save_path=args.plot_path
            )
            if plot_results:
                print(f"\n{'='*80}")
                print(f"COMPARISON SUMMARY")
                print(f"{'='*80}")
                print(f"  Memory Reduction:")
                print(f"    vs FP32 Baseline: {plot_results['memory_reduction_vs_fp32_pct']:.1f}%")
                print(f"      FP32 Baseline: {plot_results['standard_fp32']['peak_memory_mb']:.2f} MB")
                print(f"    vs FP16 Baseline: {plot_results['memory_reduction_vs_fp16_pct']:.1f}%")
                print(f"      FP16 Baseline: {plot_results['standard_fp16']['peak_memory_mb']:.2f} MB")
                print(f"      Pruned:        {plot_results['pruned']['peak_memory_mb']:.2f} MB")
                print(f"\n  Speedup:")
                print(f"    vs FP32 Baseline: {plot_results['speedup_vs_fp32']:.2f}x")
                print(f"      FP32 Baseline: {plot_results['standard_fp32']['mean_sample_latency_ms']:.3f} ms/sample")
                print(f"    vs FP16 Baseline: {plot_results['speedup_vs_fp16']:.2f}x")
                print(f"      FP16 Baseline: {plot_results['standard_fp16']['mean_sample_latency_ms']:.3f} ms/sample")
                print(f"      Pruned:        {plot_results['pruned']['mean_sample_latency_ms']:.3f} ms/sample")
                print(f"\n  Throughput Improvement:")
                print(f"    vs FP32 Baseline: {plot_results['throughput_improvement_vs_fp32_pct']:+.1f}%")
                print(f"      FP32 Baseline: {plot_results['standard_fp32']['throughput_samples_per_sec']:.2f} samples/sec")
                print(f"    vs FP16 Baseline: {plot_results['throughput_improvement_vs_fp16_pct']:+.1f}%")
                print(f"      FP16 Baseline: {plot_results['standard_fp16']['throughput_samples_per_sec']:.2f} samples/sec")
                print(f"      Pruned:        {plot_results['pruned']['throughput_samples_per_sec']:.2f} samples/sec")
            else:
                print(f"\n⚠️  Warning: Plot generation failed. Check error messages above.")
            
            # Generate forward pass memory timeline plot (with baseline overlay)
            try:
                print(f"\n[INFO] Generating forward pass memory timeline plot...")
                model_dtype = next(model.parameters()).dtype
                dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, 
                                        dtype=model_dtype).to(device)
                timeline_path = args.plot_path.replace('.png', '_timeline.png') if args.plot_path and args.plot_path.endswith('.png') else (args.plot_path + '_timeline' if args.plot_path else None)
                if timeline_path is None:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    timeline_path = f'inference_forward_pass_timeline_{timestamp}.png'
                # Create baseline (Standard ResNet-18) for overlay with CIFAR-10 config
                baseline_model = resnet18(num_classes=checkpoint_data.get('num_classes', 10)).to(device)
                # Modify for CIFAR-10: conv1 stride=1, maxpool stride=1 (matches gated/pruned model)
                # Create new layers and move them to device before replacing
                baseline_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
                baseline_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to(device)
                baseline_model.eval()
                if model_dtype == torch.float16:
                    baseline_model = baseline_model.half()
                plot_forward_pass_memory_timeline(model, dummy_input, device, save_path=timeline_path, baseline_model=baseline_model)
                del baseline_model
                clear_memory(device)
            except Exception as timeline_error:
                print(f"[WARN] Failed to generate memory timeline plot: {timeline_error}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"\n❌ Error in plot generation: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
