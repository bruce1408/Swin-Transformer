import onnx
import torch
import torch.nn as nn
from onnxsim import simplify
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import get_config
from models import build_model
from main_debug import parse_option
# from export_onnx_models.onnx_tools import convert_dynamic_to_static, onnx_simplify

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, 
                        default="/mnt/share_disk/cdd/Swin-Transformer/configs/swin/swin_tiny_patch4_window7_224.yaml",
                        # required=True, 
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=1, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/mnt/share_disk/cdd/eval_data/imagenet/", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', default="/mnt/share_disk/cdd/pretrained_models/swin_tiny_patch4_window7_224.pth", help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', 
                        default=True,
                        # action='store_true', 
                        help='Perform evaluation only')
    parser.add_argument('--throughput', 
                        # action='store_true', 
                        default=False,
                        help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", 
                        type=int, 
                        # required=True,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', 
                        # action='store_true',
                        default=False,
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', 
                        # action='store_true', 
                        default=False,
                        help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    
    parser.add_argument("--dist", default=False, help="describe if use multi distribution to train/val")
    
    parser.add_argument("--onnx-dir", default="/mnt/share_disk/cdd/export_onnx_models", help="onnx path")
    parser.add_argument("--onnx-path", default="/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_20240201_194422_swin_repo.onnx", help="onnx path")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    
    model = build_model(config)
    # with torch.no_grad():
    model.eval()
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出模型到 ONNX
    now = datetime.datetime.now()
    onnx_path = os.path.join(args.onnx_dir, f"swin_tiny_patch4_window7_224_{now.strftime('%Y%m%d_%H%M%S')}_swin_repo.onnx")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        # dynamic_axes={'input': {0: 'batch_size'}, 
        #                 'output': {0: 'batch_size'}}
    )



def export_onnx_manual():
    args, config = parse_option()
    model = build_model(config)
    checkpoint = torch.load('/mnt/share_disk/cdd/pretrained_models/swin_tiny_patch4_window7_224.pth', map_location='cpu')
    print('start eval')
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    input_names = ['input']
    output_names = ['output']

    print('start export')
    now = datetime.datetime.now()
    # onnx_path  = f"swim_transformer_{now.strftime('%Y%m%d_%H%M%S')}.onnx"
    
    onnx_path  = f"/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_{now.strftime('%Y%m%d_%H%M%S')}_swin_repo.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=17,
        do_constant_folding=True, 
        input_names=input_names, 
        output_names=output_names
    )
    
    
if __name__ == '__main__':
    args, config = parse_option()
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    now = datetime.datetime.now()  
         
    # onnx_path = os.path.join(args.onnx_dir, f"swin_tiny_patch4_window7_224_{now.strftime('%Y%m%d_%H%M%S')}_swin_repo.onnx")
    
    main(config, args)
    # export_onnx_manual()
    
    # convert_dynamic_to_static(args.onnx_path)
    # onnx_simplify(args.onnx_path)
