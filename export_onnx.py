import torch
import argparse
import importlib
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Export OSTrack model to ONNX')
    parser.add_argument('--script', type=str, default='ostrack', help='script name')
    parser.add_argument('--config', type=str, default='ostrack', help='yaml configure file name')
    parser.add_argument('--output', type=str, default='ostrack.onnx', help='output onnx file name')
    parser.add_argument('--pretrained', type=str, default='vitb_384_mae_ce_32x4_got10k_ep100/OSTrack_ep0100.pth.tar', help='pretrained model path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    yaml_fname = 'experiments/ostrack/vitb_384_mae_ce_32x4_got10k_ep100.yaml'
    config_module = importlib.import_module(f'lib.config.{args.script}.config')
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    
    # Create model
    model_module = importlib.import_module('lib.models')
    model_constructor = model_module.build_ostrack
    model = model_constructor(cfg, training=False)
    
    # Load pretrained weights
    pretrained_path = os.path.join(args.pretrained)
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded pretrained model from {pretrained_path}')
    else:
        raise FileNotFoundError(f'Pretrained model not found at {pretrained_path}')
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy inputs
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    template = torch.randn(bs, 3, z_sz, z_sz)
    search = torch.randn(bs, 3, x_sz, x_sz)
    
    # Export to ONNX
    torch.onnx.export(
        model,                     # model being run
        (template, search),        # model input (tuple of tensors)
        args.output,              # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=12,         # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['template', 'search'],   # the model's input names
        output_names=['pred_boxes', 'score_map'],  # the model's output names
        dynamic_axes={
            'template': {0: 'batch_size'},
            'search': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'},
            'score_map': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {args.output}")

if __name__ == "__main__":
    main() 