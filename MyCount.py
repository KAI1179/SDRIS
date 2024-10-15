import cv2
import torch
import utils.config as config
from model import build_segmenter
from utils.dataset import tokenize
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format
from torchvision.models import resnet50
from torchstat import stat
# input_size = (416, 416)

def getTransformMat(img_size, inverse=False):
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None

def convert(img):
    img_size = img.shape[:2]
    mat, mat_inv = getTransformMat(img_size, False)
    img = cv2.warpAffine(
        img,
        mat,
        input_size,
        flags=cv2.INTER_CUBIC,
        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
    pad_img = img
    
    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()

    mean = torch.tensor([0.48145466, 0.4578275,
                         0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                        0.27577711]).reshape(3, 1, 1)
    img.div_(255.).sub_(mean).div_(std)
    return img, pad_img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = config.load_cfg_from_cfg_file("./config/refcoco/cris_r50.yaml")
    PATH = "exp/refcoco/RN50_32_60e_base_Fv4/best_model.pth"
    # {CRIS_Phrase_5_Area_200_layer4, RN101_32_60e_allFusionFs_CRIS_Phrase_5_Area_200_layer4_ks3
    # CRIS_Area_200, RN50_32_60e_allFusionFs_CRIS_Phrase_5_Area_200_layer4_ks3}
    model, _ = build_segmenter(cfg)
    model.to(device)
    print('model device:', next(model.parameters()).device)
    model = torch.nn.DataParallel(model)

    # checkpoint = torch.load(PATH, map_location=torch.device('cuda:0'))
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    for name, parameters in model.named_parameters():
        if parameters.device == 'cpu':
            print(name, ':', parameters.device)


    print('model device:', next(model.parameters()).device)
    print("=> loaded checkpoint '{}'".format(PATH))


    # img = cv2.imread("./Vis/COCO_train2014_000000027950.jpg")
    # # COCO_train2014_000000010471; COCO_train2014_000000027950; COCO_train2014_000000003478
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # resized_img = cv2.resize(img, (416, 416), interpolation= cv2.INTER_LINEAR)
    # # img_ndarray = np.array(img)
    # # original_h, original_w = img_ndarray.shape[0], img_ndarray.shape[1]
    # img, pad_img = convert(img)
    # img = img.unsqueeze(0)
    # # print('img shape: ', img.shape)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # img = img.to(device)
    # print('img device: ', img.device)

    # sent = "guy in hat"   # small elephant; guy in hat; person carrying black bag; cow behind the tree
    # text = tokenize(sent, 17, True)
    
    # text = text.cuda(non_blocking=True)
    # pred = model(img, text)

    # flops, params = get_model_complexity_info(model, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print("%s |%s |%s" % (model_name, flops, params))

    in_img = torch.randn(1, 3, 416, 416)
    # stat(model_test, (3, 416, 416))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # in_img = in_img.to(device)
    # print('in_img device: ', in_img.device)
    # flops, params = profile(model_test, inputs=(in_img, ), verbose=True)
    # flops, params = clever_format([flops, params], "%.3f")
    # print("FLOPs: ", flops)
    # print("Param: ", params)
    
    # 分析FLOPs
    flops = FlopCountAnalysis(model, in_img)
    print("FLOPs: ", flop_count_table(flops))
    # 分析parameters
    # print("Param: ", parameter_count_table(model))
