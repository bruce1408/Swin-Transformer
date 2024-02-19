import os
import onnx
import torch
import argparse
from tqdm import tqdm
import onnxruntime
import numpy as np
import PIL
import datetime
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# 1. load config variable and load model from pytorch
# def parse_args():
#     parser = argparse.ArgumentParser(description='PyTorch convert onnx to verficaiton Acc')
    
#     # 模型选择
#     parser.add_argument('--model_name', default="resnet18", help='model name resnet-50、mobilenet_v2、efficientnet')
    
#     # 预处理模型下载地址
#     parser.add_argument('--model_file', default="/root/resnet18-f37072fd.pth", help='laod checkpoints from saved models')
    
#     # 输入大小
#     parser.add_argument('--input_shape', type=list, nargs='+', default=[1,3,224,224])

#     # label 地址
#     parser.add_argument("--label_path", default="/root/val_torch_cls/synset.txt", help="label path")
    
#     # onnx 输出地址
#     parser.add_argument('--export_path', default="/mnt/share_disk/cdd/Swin-Transformer/swim_transformer_20240130_193406.onnx", help="pth model convert to onnx name")
    
#     # 验证集 地址
#     parser.add_argument('--data_val', default="/mnt/share_disk/cdd/eval_data/imagenet/val", help="val data")
    

#     args = parser.parse_args()

#     return args


# img preprocessing method
# def pre_process_img(image_path):
    
#     transform = transforms.Compose([
#         # transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], 
#             std=[0.229, 0.224, 0.225])
#     ])
    
#     img = PIL.Image.open(image_path).convert("RGB")
#     img = transform(img)
#     return img


# # get the val data and preprocessing 
# def load_label_imgs():
#     print("laoding the data and preprocessing the image data ...")
    
#     count = 0 
#     correct_count = 0
#     dir_name_list = sorted(os.listdir(args.data_val))
#     sess = onnxruntime.InferenceSession(args.export_path)
#     input_name = sess.get_inputs()[0].name

#     for true_label, dir_ in enumerate(dir_name_list):
#         img_dir = os.path.join(args.data_val, dir_)
#         # print(img_dir)
#         for img_name in os.listdir(img_dir):
#             count += 1
#             img_data = pre_process_img(os.path.join(img_dir, img_name))
#             input_data = np.expand_dims(img_data, axis=0).astype(np.float32)
#             output = sess.run(None, {input_name: input_data})
            
#             predicted_class = np.argmax(output)
#             print(predicted_class)
#             if predicted_class == true_label:
#                 correct_count += 1
#         if count % 100 == 0 :
#             print("the acc is {}".format(correct_count / count))
            
#     accuracy = correct_count / count
#     print('Classification accuracy:', accuracy)

    


# # get the torch official models 
# def get_model():
#     model = models.__dict__[args.model_name](pretrained=False)
#     state_dict = torch.load(args.model_file)
#     model.load_state_dict(state_dict)
#     return model


# # export .pth model to .onnx
# def export_onnx(model, input_shape, export_onnx_path):
#     model = get_model()
#     model.eval()
#     torch.onnx.export(model, torch.randn(input_shape), export_onnx_path, input_names=["input"], output_names=["output"], opset_version=11)
#     print("onnx model has been transformed!")


# # load onnx and val the data with ort
# def load_onnx_and_eval():
#     sess = onnxruntime.InferenceSession(args.export_path)
#     correct_count = 0
#     total_count = len(os.listdir(args.data_val))
    
#     print("begin to eval the model...")
#     for i in tqdm(range(total_count)):
#         input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
#         output = sess.run(None, {'img': input_data})[0]
#         predicted_class = np.argmax(output)
#         # predict = img_dict[predicted_class]
    
#         if predicted_class == img_labels[i]:
#             correct_count += 1
            
#     accuracy = correct_count / total_count
#     print('Classification accuracy:', accuracy)


# if __name__ == "__main__":
#     args = parse_args()
    
#     # 1. 加载数据
#     load_label_imgs()
#     # print(image_datas)
#     # print(img_labels)
    
    
#     # 2. 加载模型
#     # model = get_model()
    
#     # 3. 导出onnx模型
#     # export_onnx(model, args.input_shape, args.export_path)
    
#     # 4. 用ort进行推理验证
#     # load_onnx_and_eval(args)
    
   
   
   
# import os
import json
import tqdm
from PIL import Image
import onnxruntime
from torchvision import transforms
import matplotlib.pyplot as plt
from termcolor import cprint, colored
from config import get_config
from models import build_model

        
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
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
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
                        action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, 
                        # required=True,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    
    parser.add_argument("--dist", default=False, help="describe if use multi distribution to train/val")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

   
def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)
        

def print_colored_box(text, text_color='white', box_color='green'):
    
    bold_start = "\033[1m"
    bold_end = "\033[0m"
    
    # 测量文本长度，并为方框的左右添加空格
    padded_text = " " + text + " "
    text_length = len(padded_text)
    
    # 生成上下边框
    top_bottom_border = '+' + '-' * text_length + '+'
    # 为边框添加颜色
    # colored_top_bottom = colored(top_bottom_border, box_color)
    colored_top_bottom = colored(bold_start + top_bottom_border + bold_end, box_color)

    # 生成中间文本行，包括左右边框
    middle_line = "|" + padded_text + "|"

    # 为文本和边框添加颜色
    # colored_middle = colored(middle_line, text_color, 'on_' + box_color)
    # colored_middle = colored(middle_line, text_color, attrs=['bold'])
    colored_middle_text = colored(text, text_color, attrs=['bold'])

    colored_middle = bold_start + colored("|", box_color) + colored_middle_text + colored("|", box_color) + bold_end


    # colored_text = colored(text, color, attrs=['bold'])

    # 打印彩色方框
    print(colored_top_bottom)
    print(colored("|", box_color) + bold_start + " " + bold_end + colored_middle_text + bold_start + " " + bold_end + colored("|", box_color))
    print(colored_top_bottom)


    

def model_convert_onnx(model, input_shape, output_path):
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
    input_names = ["input1"]
    output_names = ["output1"]      

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=12,       # 版本通常为10 or 11
        input_names=input_names,
        output_names=output_names
    )

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # onnx_path = '/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_20240202_152404_swin_repo_1.onnx'
    onnx_path = "/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_20240219_162237_swin_repo_1.onnx"
    args, config = parse_option()
    model = build_model(config)
    # model = create_model(num_classes=1000).to(device)
    # load model weights
    # model_weight_path = "/mnt/share_disk/cdd/pretrained_models/swin_tiny_patch4_window7_224.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    count = 0
    total_count = 0
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
        # [transforms.Resize(256),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    eval_dir_path = "/mnt/share_disk/cdd/eval_data/imagenet/val"
    img_dirs_list = sorted(os.listdir(eval_dir_path))
    for label_truth, img_dir in (enumerate(img_dirs_list)):
        img_dir_path = os.path.join(eval_dir_path, img_dir)
        img_name_lists = os.listdir(img_dir_path)
        for img_name in img_name_lists:
            img_path = os.path.join(img_dir_path, img_name)

            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
                
            #---------------------------------------------------------#
            #   使用 onnxruntime 进行推理
            #---------------------------------------------------------#
            image_data = img.numpy()
            sess_options = onnxruntime.SessionOptions()
            # print(onnxruntime.get_device())
            # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_AL
            # ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
            
            # # 注意这儿的 input1 需要和model_convert_onnx()中定义的模型输入名称相同！
            ort_inputs = {"image": image_data}
            onnx_outputs = ort_session.run(None, ort_inputs)

            output = torch.from_numpy(onnx_outputs[0])
            output = torch.squeeze(output).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            if (label_truth == predict_cla):
                count+=1
            total_count += 1
        if total_count % 50 == 0 :
            print("Acc is %.3f "%(count / total_count))
            break
    
    print_info('------------------------------------------------------------------------------\n'
           '|                    ImageNet Dataset Evaluation Results                      |\n'
           '|                                                                             |\n'
           f'| the total img nums is {total_count}, the right predict num is {count}, acc is: {count / total_count:.3}         |\n'
           '------------------------------------------------------------------------------', ['yellow', 'bold'])
    print_colored_box(f'the total img nums is {total_count}, the right predict num is {count}, acc is: {count / total_count:.3}', 'yellow', 'yellow')
    # print_colored_box('Hello, World!', 'yellow', 'red')


def print_colored_box_line(title, message, box_color='yellow', text_color='white'):
    # 定义方框的宽度为终端的宽度，这里假定为80字符宽
    box_width = 80
    
    # 创建顶部和底部的边框
    horizontal_border = '+' + '-' * (box_width - 2) + '+'
    colored_horizontal_border = colored(horizontal_border, box_color)
    
    # 创建标题和消息文本，使其居中
    title_text = f"| {title.center(box_width - 4)} |"
    message_text = f"| {message.center(box_width - 4)} |"
    
    # 添加颜色到文本
    colored_title = colored(title_text, text_color, 'on_' + box_color)
    colored_message = colored(message_text, text_color, 'on_' + box_color)
    
    # 打印方框
    print(colored_horizontal_border)
    print(colored_title)
    print(colored_horizontal_border)
    print(colored_message)
    print(colored_horizontal_border)




if __name__ == '__main__':
    # print_colored_box(' Hello, World! ')
    # print_colored_box('Hello, World!', 'blue')
    # print_colored_box('Hello, World!', 'yellow', 'red')

    main()

    # 使用示例
    # title = "ImageNet Dataset Evaluation Results"
    # message = "the total img nums is 50, the right predict num is 46, acc is: 0.92"
    # print_colored_box_line(title, message)
