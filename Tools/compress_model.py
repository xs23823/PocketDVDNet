import torch
import torch.nn as nn
from models.fastdvdnet import FastDVDnet
import copy

def get_channel_importance(conv_layer, dim='out'):
    """Calculate channel importance based on L1 norm of weights"""
    weights = conv_layer.weight.data
    
    if dim == 'out':
        importance = weights.abs().sum(dim=(1, 2, 3))
    else:  # dim == 'in'
        importance = weights.abs().sum(dim=(0, 2, 3))
    
    return importance

def important_channels(importance, n_keep):
    """Select the most important channels"""
    if n_keep >= len(importance):
        return torch.arange(len(importance))
    
    _, indices = torch.topk(importance, n_keep)
    return indices.sort()[0]

def compress_conv(old_conv, new_in_channels, new_out_channels):
    """Compress a conv layer by selecting most important channels"""
    
    # importance scores
    out_importance = get_channel_importance(old_conv, 'out')
    in_importance = get_channel_importance(old_conv, 'in')
    
    out_indices = important_channels(out_importance, new_out_channels)
    in_indices = important_channels(in_importance, new_in_channels)
    
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=new_out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None)
    )
    
    # copy selected weights
    old_weight = old_conv.weight.data
    new_weight = old_weight[out_indices][:, in_indices]
    new_conv.weight.data = new_weight
    
    # copy selected bias
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data[out_indices]
    
    return new_conv, out_indices

def compress_bn(old_bn, out_indices):
    """Compress batch norm layer based on selected output channels"""
    if old_bn is None:
        return None
    
    new_bn = nn.BatchNorm2d(
        num_features=len(out_indices),
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats
    )
    
    if old_bn.weight is not None:
        new_bn.weight.data = old_bn.weight.data[out_indices]
        new_bn.bias.data = old_bn.bias.data[out_indices]
        new_bn.running_mean.data = old_bn.running_mean.data[out_indices]
        new_bn.running_var.data = old_bn.running_var.data[out_indices]
    
    return new_bn

def compress(model, layer_specs):
    """Apply compression based on specs inputted"""
    
    compressed_model = copy.deepcopy(model)
    
    for layer_path, (new_c_in, new_c_out) in layer_specs.items():
        print(f"Compressing {layer_path}: → {new_c_in}, {new_c_out}")
        
        # navigate to the conv layer
        parts = layer_path.split('.')
        module = compressed_model
        for part in parts[:-1]:
            module = getattr(module, part)
        
        old_conv = getattr(module, parts[-1])
        orig_in, orig_out = old_conv.in_channels, old_conv.out_channels
        print(f"  Original: {orig_in} → {orig_out}")
        
        new_conv, out_indices = compress_conv(old_conv, new_c_in, new_c_out)
        setattr(module, parts[-1], new_conv)
        
        try:
            # FastDVDnet pattern: conv layer N is followed by BN layer N+1
            conv_num = int(parts[-1])
            bn_num = conv_num + 1
            
            parent_module = compressed_model
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            if hasattr(parent_module, str(bn_num)):
                old_bn = getattr(parent_module, str(bn_num))
                if isinstance(old_bn, nn.BatchNorm2d):
                    new_bn = compress_bn(old_bn, out_indices)
                    setattr(parent_module, str(bn_num), new_bn)
                    print(f"  → Also compressed BN layer")
        except:
            pass  # no BN layer or cudnt find it
    
    return compressed_model

def load_n_squish(model_path, layer_specs):
    """Load sparse model and apply compression"""
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    original_model = FastDVDnet(num_input_frames=5, num_color_ch=3)
    original_model.load_state_dict(checkpoint['model_state'])
    original_model.eval()
    
    #  original parameters
    original_params = sum(p.numel() for p in original_model.parameters())
    print(f"Original model: {original_params:,} parameters")
    
    compressed_model = compress(original_model, layer_specs)
    
    #  compressed parameters
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    reduction = (1 - compressed_params/original_params) * 100
    
    print(f"\nCompression Results:")
    print(f"Original: {original_params:,} parameters")
    print(f"Compressed: {compressed_params:,} parameters")
    print(f"Reduction: {reduction:.1f}%")
    
    return compressed_model

def test(compressed_model, layer_specs=None):
    
    print("\nTesting...")
    
    try:
        with torch.no_grad():
            x = torch.randn(1, 15, 96, 96)
            noise_map = torch.randn(1, 15, 96, 96)
            output = compressed_model(x, noise_map)
            
            print(f"Forward pass successful!")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            return True
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def main():
    # define compression specifications
    # Format: 'layer_path': (new_input_channels, new_output_channels)
    layer_specs = {
        'temp1.inc.convblock.0': (18, 90),
        'temp1.inc.convblock.3': (90, 16),
        'temp1.downc0.convblock.0': (16, 32),
        'temp1.downc0.convblock.3.convblock.0': (32, 32),
        'temp1.downc0.convblock.3.convblock.3': (32, 32),
        'temp1.downc1.convblock.0': (32, 64),
        'temp1.downc1.convblock.3.convblock.0': (64, 64),
        'temp1.downc1.convblock.3.convblock.3': (64, 64),
        'temp1.upc2.convblock.0.convblock.0': (64, 64),
        'temp1.upc2.convblock.0.convblock.3': (64, 64),
        'temp1.upc2.convblock.1': (64, 128),
        'temp1.upc1.convblock.0.convblock.0': (32, 32),
        'temp1.upc1.convblock.0.convblock.3': (32, 32),
        'temp1.upc1.convblock.1': (32, 64),
        'temp1.outc.convblock.0': (16, 32),
        'temp1.outc.convblock.3': (32, 3),
        
        'temp2.inc.convblock.0': (18, 90),
        'temp2.inc.convblock.3': (90, 16),
        'temp2.downc0.convblock.0': (16, 32),
        'temp2.downc0.convblock.3.convblock.0': (32, 32),
        'temp2.downc0.convblock.3.convblock.3': (32, 32),
        'temp2.downc1.convblock.0': (32, 64),
        'temp2.downc1.convblock.3.convblock.0': (64, 64),
        'temp2.downc1.convblock.3.convblock.3': (64, 64),
        'temp2.upc2.convblock.0.convblock.0': (64, 64),
        'temp2.upc2.convblock.0.convblock.3': (64, 64),
        'temp2.upc2.convblock.1': (64, 128),
        'temp2.upc1.convblock.0.convblock.0': (32, 32),
        'temp2.upc1.convblock.0.convblock.3': (32, 32),
        'temp2.upc1.convblock.1': (32, 64),
        'temp2.outc.convblock.0': (16, 32),
        'temp2.outc.convblock.3': (32, 3),
    }
    
    model_path = '/home/imogend/Documents/Data/out_extra_data/best_sparsity.pt'
    
    try:
        compressed_model = load_n_squish(model_path, layer_specs)
        
        if test(compressed_model, layer_specs):
            torch.save({
                'model_state_dict': compressed_model.state_dict(),
                'layer_specs': layer_specs, 
                'model_config': {'num_input_frames': 5, 'num_color_ch': 3},
            }, 'compressed_fastdvdnet1.pt')
        else:
            print("Model compression failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()