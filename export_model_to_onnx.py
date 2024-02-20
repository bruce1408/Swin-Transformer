import sys
import torch
import datetime
from collections import OrderedDict
import warnings

# 忽略所有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# sys.path.append('./Swin-Transformer')
from models import build_model
from main_debug import parse_option


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def model_transfer(args, config):
    batch_size = config.DATA.BATCH_SIZE
    print(batch_size)
    onnx_model = build_model(config)
    ckpt = torch.load(config.MODEL.RESUME, map_location='cpu')
    ckpt['model'] = proc_nodes_module(ckpt, 'model')
    onnx_model.load_state_dict(ckpt['model'])
    onnx_model.cpu()
    input_names = ["image"]
    output_names = ["class"]
    dummy_input = torch.randn(batch_size, 3, 384, 384)
    fp16 = False
    # if fp16:
    #     onnx_path = 'models/onnx/swin_tiny_bs' + str(batch_size) + '_fp16.onnx'
    # else:
    #     onnx_path = 'models/onnx/swin_tiny_bs' + str(batch_size) + '.onnx'
    
    now = datetime.datetime.now()
    # onnx_path  = f"/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_{now.strftime('%Y%m%d_%H%M%S')}_swin_repo_" + str(batch_size)+ "_"+str(args.opset)+".onnx"
    onnx_path  = f"/mnt/share_disk/cdd/export_onnx_models/swin_large_patch4_window12_384_{now.strftime('%Y%m%d_%H%M%S')}_swin_repo_" + str(batch_size)+ "_"+str(args.opset)+".onnx"
    
    torch.onnx.export(
        onnx_model, 
        dummy_input, 
        onnx_path, 
        input_names=input_names,
        output_names=output_names, 
        opset_version=17, 
        verbose=False
        )
    print("model saved successful at ", onnx_path)


if __name__ == '__main__':
    args, main_config = parse_option()
    model_transfer(args, main_config)