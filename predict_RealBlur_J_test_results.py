from __future__ import print_function
import numpy as np
import torch
import cv2
import yaml
import os
from torch.autograd import Variable
from models.networks import get_generator
import torchvision
import time
import torch.nn.functional as F
import argparse

def get_args():
    parser = argparse.ArgumentParser('Test an image')
    parser.add_argument('--job_name', default='fsformer_without_fs',
    type=str, help='current job s name')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open(os.path.join('config/', args.job_name, 'config_stage2.yaml')) as cfg:
    #with open(os.path.join('config/', args.job_name, 'config_pretrained.yaml')) as cfg:
        config = yaml.safe_load(cfg)
    blur_path = '/scratch/user/hanzhou1996/datasets/deblur/RealBlur_J/test/testA'
    out_path = os.path.join('results', args.job_name, 'images_realj')
    weights_path = os.path.join('results', args.job_name, 'models', 'final_XYScanNet_stage2.pth')  # change the model name to test different phases: final/best final_StripMamba_pretrained.pth
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(weights_path))
    model = model.cuda()
    test_time = 0
    iteration = 0
    total_image_number = 980

    # warm up
    warm_up = 0
    print('Hardware warm-up')
    for file in os.listdir(blur_path):
        #if not os.path.isdir(out_path + '/' + file):
        #    os.mkdir(out_path + '/' + file)
        img_name = file
        # for img_name in os.listdir(blur_path + '/' + file):
        warm_up += 1
        img = cv2.imread(blur_path + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
        with torch.no_grad():
            img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
            factor = 8
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            result_image, decomp1, decomp2 = model(img_tensor)
            #result_image = model(img_tensor)
        if warm_up == 20:
            break
        break

    for file in os.listdir(blur_path):
        #if not os.path.isdir(out_path + '/' + file):
        #    os.mkdir(out_path + '/' + file)
        img_name = file
        # for img_name in os.listdir(blur_path + '/' + file):
        img = cv2.imread(blur_path + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
        with torch.no_grad():
            iteration += 1
            img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

            factor = 8
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            H, W = img_tensor.shape[2], img_tensor.shape[3]

            start = time.time()
            _output, decomp1, decomp2 = model(img_tensor)
            #_output = model(img_tensor)
            stop = time.time()

            result_image = _output[:, :, :h, :w]
            result_image = torch.clamp(result_image, -0.5, 0.5)
            result_image = result_image + 0.5

            test_time += stop - start
            print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
            print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
            out_file_name = out_path + '/' + img_name 
            torchvision.utils.save_image(result_image, out_file_name)

