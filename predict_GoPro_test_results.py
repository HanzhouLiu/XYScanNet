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
import argparse
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser('Test an image')
    parser.add_argument('--job_name', default='xyscannet',
    type=str, help='current job s name')
    return parser.parse_args()

def print_max_gpu_usage():
    """Prints the maximum GPU memory usage in GB."""
    max_memory = torch.cuda.max_memory_allocated()
    max_memory_in_gb = max_memory / (1024 ** 3)  # Convert bytes to GB
    print(f"Maximum GPU memory usage during test: {max_memory_in_gb:.2f} GB")

if __name__ == '__main__':
    # optionally reset gpu
    #torch.cuda.reset_max_memory_allocated()
    args = get_args()
    #with open(os.path.join('config/', args.job_name, 'config_stage2.yaml'), 'r') as cfg:  # change the CFG name to test different models: pretrained, gopro, refined, stage1, stage2
    #    config = yaml.safe_load(cfg)
    with open(os.path.join('config/', args.job_name, 'config_stage2.yaml'), 'r') as cfg:  # change the CFG name to test different models: pretrained, gopro, refined, stage1, stage2
        config = yaml.safe_load(cfg)
    blur_path = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GOPRO_/test/testA'
    out_path = os.path.join('results', args.job_name, 'images')
    weights_path = os.path.join('results', args.job_name, 'models', 'best_{}.pth'.format(config['experiment_desc']))  # change the model name to test different phases: final/best
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(weights_path))
    model = model.cuda()
    #model.eval()

    test_time = 0
    iteration = 0
    total_image_number = 1111

    # warm-up
    warm_up = 0
    print('Hardware warm-up')
    for file in os.listdir(blur_path):
        for img_name in os.listdir(blur_path + '/' + file):
            warm_up += 1
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
                result_image, decomp1, decomp2 = model(img_tensor)
                #result_image = model(img_tensor)
            if warm_up == 20:
                break
        break

    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()

                start = time.time()
                result_image, decomp1, decomp2 = model(img_tensor)
                #result_image = model(img_tensor)
                stop = time.time()
                print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
                test_time += stop - start
                print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
                result_image = result_image + 0.5
                out_file_name = out_path + '/' + file + '/' + img_name
                # optionally save image
                torchvision.utils.save_image(result_image, out_file_name)
    
    # optionally print gpu usage
    #print_max_gpu_usage()
    #torch.cuda.reset_max_memory_allocated()