import argparse
import os
import sys
from PIL import Image
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision import transforms
from DeepLabV3Plus import network
from DeepLabV3Plus.datasets import VOCSegmentation, Cityscapes

sys.path.append('./pytorch_pix2pix')
from models.pix2pix_model import Pix2PixModel

import numpy as np
from skimage.transform import resize


def transfer_to_color_map(label_img, opts):
    if opts.dataset == 'cityscapes':
        return Cityscapes.decode_target(label_img)
    elif opts.dataset == 'voc':
        return VOCSegmentation.decode_target(label_img)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='B', choices=['A', 'B'],  help='')

    parser.add_argument('--singan_mode', type=str, help='', default='random_samples')
    # parser.add_argument('--singan_ref_mode_name', type=str, help='', default=None)
    parser.add_argument('--singan_model', type=str, help='', default='aachen')

    opts = parser.parse_args()

    if opts.mode == 'A':
        sys.path.append('./SinGAN')
        from SinGAN.SinGAN import functions
        from SinGAN.SinGAN.manipulate import SinGAN_generate
    elif opts.mode == 'B':
        sys.path.append('./SinGAN_B_Upsample')
        from SinGAN import functions
        from SinGAN.manipulate import SinGAN_generate_V3
    else:
        raise Exception('Unsupport method mode: %s' % opts.mode)

    singan_opt = Namespace(min_size=25, max_size=256, scale_factor=0.7722260524731895, nc_im=3, not_cuda=False,
                               mode='random_samples', alpha=10, gen_start_scale=0, manualSeed=None,
                               input_name='aaaaa.png', out='SinGAN_Output', ker_size=3, num_layer=5, nc_z=3,
                               niter=2000, noise_amp=0.1, nfc=32, min_nfc=32,  scale_factor_init=0.75)
    singan_opt = functions.post_config(singan_opt)

    seg_img = Image.open('aachen_gt.png').convert('RGB')
    seg_img = seg_img.resize((256, 256), Image.NEAREST)
    seg_img = functions.np2torch(np.array(seg_img), singan_opt)

    singan_dir = 'SinGAN_B_Upsample/TrainedModels/%s/scale_factor=%f_seg' % (
            opts.singan_model, 0.75)
    originals = torch.load(os.path.join(singan_dir, 'segs.pth'))
    h, w = originals[-1].shape[2:]
    seg_imgs = []

    for i, img in enumerate(originals):
        h, w = img.shape[2:]

        curr_seg = resize(seg_img.cpu().numpy(), (1, 3, h, w), order=0)
        seg_imgs.append(torch.from_numpy(curr_seg).type(torch.cuda.FloatTensor))

    if not os.path.exists(singan_opt.out):
        os.mkdir(singan_opt.out)

    del originals
        # functions.adjust_scales2image(fake_tensor, singan_opt)

    Gs = torch.load(os.path.join(singan_dir, 'Gs.pth'))
    Zs = torch.load(os.path.join(singan_dir, 'Zs.pth'))

        # reals = functions.creat_reals_pyramid(fake_tensor,fake_imgs,singan_opt)
        # reals = torch.load(os.path.join(singan_dir, 'reals.pth'))
    NoiseAmp = torch.load(os.path.join(singan_dir, 'NoiseAmp.pth'))
    in_s = functions.generate_in2coarsest(seg_imgs, 1, 1, singan_opt)
    SinGAN_generate_V3(Gs, Zs, seg_imgs, NoiseAmp, singan_opt, None, num_samples=50)

main()