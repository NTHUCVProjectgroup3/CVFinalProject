import argparse
import os
import sys
import shutil
from PIL import Image
from argparse import Namespace
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms

# from torchvision import transforms
# from DeepLabV3Plus import network
# from DeepLabV3Plus.datasets import VOCSegmentation, Cityscapes

sys.path.append('./pytorch_pix2pix')
from models.pix2pix_model import Pix2PixModel

from skimage.transform import resize
from skimage.measure import compare_ssim

import cv2

def transfer_to_color_map(label_img, opts):
    if opts.dataset == 'cityscapes':
        return Cityscapes.decode_target(label_img)
    elif opts.dataset == 'voc':
        return VOCSegmentation.decode_target(label_img)

def psnr(src, dst):
    mse = np.mean( (src - dst) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_picture', type=str, help='select one image from your disk', required=True)
    parser.add_argument('--source_picture', type=str, help='', default=None)
    parser.add_argument('--mode', type=str, default='A', choices=['A', 'B'],  help='')
    parser.add_argument('--size', type=int, help='', default=256)

    parser.add_argument('--pix2pix_path', type=str, help='select pix2pix pretrained model path', default=None)

    parser.add_argument('--singan_model', type=str, help='select one trained singan model', default=None)
    parser.add_argument('--singan_mode', type=str, help='', default='random_samples')
    parser.add_argument('--singan_model_dir', type=str, help='', default='SinGAN_C')
    # parser.add_argument('--singan_ref_mode_name', type=str, help='', default=None)
    
    parser.add_argument('--sample_size', type=int, help='', default=1)
    parser.add_argument('--manualSeed', type=int, help='', default=None)

    opts = parser.parse_args()
    opts.singan_model_dir = opts.singan_model_dir.replace('/', '')

    '''
    img = cv2.imread('ulm_fake.png')# '%s/%s_%s/%s.png' % ('SinGAN_Output', opts.singan_model, 'D', opts.input_picture[:-4]))
    # print('%s/%s_%s/%s.png' % ('SinGAN_Output', opts.singan_model, 'A', opts.input_picture[:-4]))
    # print(os.path.exists('%s/%s_%s/%s.png' % ('SinGAN_Output', opts.singan_model, 'A', opts.input_picture[:-4])))
    if opts.source_picture is not None:
        src = cv2.imread(opts.source_picture)
        src = cv2.resize(src, (256,256))
        print('PSNR: %f' % psnr(src, img))
        print('SSIM: %f' % compare_ssim(src, img,  multichannel=True))
        exit(0)
    '''
    seg_img = Image.open(opts.input_picture).convert('RGB')
    seg_img_ = seg_img.resize((opts.size,opts.size), Image.NEAREST)        

    model_dirs = []
    for model_dir in opts.singan_model_dir.split(','):
        model_dirs.append(model_dir.strip())

    
    if opts.mode == 'A':
        if opts.pix2pix_path is None:
            raise Exception('Error pix2pix_path is empty!')

        transform2 = transforms.Compose([
            # transforms.Resize((286,286), Image.NEAREST),
            # transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        pix2pix_opt = Namespace(gpu_ids=[0], isTrain=False, preprocess='resize_and_crop', checkpoints_dir='aa', name='bb',
                                input_nc=3, output_nc=3, ngf=64, netD='basic', n_layers_D=3, norm='batch',
                                init_type='normal', no_dropout=False, init_gain=0.02, netG='unet_256')
        pix2pix_model = Pix2PixModel(pix2pix_opt)
        netG_pix2pix_state = torch.load(opts.pix2pix_path)
        pix2pix_model.netG.module.load_state_dict(netG_pix2pix_state)
        pix2pix_model.eval()

        with torch.no_grad():
            tensor_img2 = transform2(seg_img_)
            tensor_img2 = torch.unsqueeze(tensor_img2, 0)
            pix2pix_model.real_A = tensor_img2
            pix2pix_model.forward()

            fake = pix2pix_model.fake_B.data
            fake = fake[0].cpu().float().numpy()

            fake = (np.transpose(fake, (1, 2, 0)) + 1) / 2.0 * 255.0
            fake = fake.astype(np.uint8)

        fake_img = Image.fromarray(fake)
        fake_img.save('%s_fake.png' % opts.input_picture[:-4])
        if opts.source_picture is not None:
            src = cv2.imread(opts.source_picture)
            src = cv2.resize(src, (opts.size,opts.size))
            print('PSNR: %f' % psnr(src, fake[..., ::-1]))
            print('SSIM: %f' % compare_ssim(src, fake[..., ::-1],  multichannel=True))
        return

    if opts.singan_model is None:
        raise Exception("aaaa")

    for model_dir in model_dirs:
        # print(sys.path)
        sys.path.append(model_dir)
        # import SinGAN
        from SinGAN import functions
        from SinGAN.manipulate import SinGAN_generate_V3
        # print(SinGAN.__path__)

        
        singan_opt = Namespace(min_size=25, max_size=opts.size, scale_factor=0.7722260524731895, nc_im=3, not_cuda=False,
                               mode=opts.singan_mode, alpha=10, gen_start_scale=0, manualSeed=None,
                               input_name=opts.input_picture, out='SinGAN_Output', ker_size=3, num_layer=5, nc_z=3,
                               niter=2000, noise_amp=0.1, nfc=32, min_nfc=32)

        if opts.manualSeed:
            singan_opt.manualSeed = opts.manualSeed


        singan_opt = functions.post_config(singan_opt)
        seg_img = functions.np2torch(np.array(seg_img_), singan_opt)
        
        singan_dir = '%s/TrainedModels/%s/scale_factor=%f_seg' % (model_dir, opts.singan_model, 0.75)
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
        SinGAN_generate_V3(Gs, Zs, seg_imgs, NoiseAmp, singan_opt, None, num_samples=opts.sample_size)

        out_dir = '%s/RandomSamples/%s/gen_start_scale=%d' % (singan_opt.out, opts.input_picture[:-4], singan_opt.gen_start_scale)
        if not os.path.exists('%s/%s_%s' % (singan_opt.out, opts.singan_model, model_dir[-1])):
            os.mkdir('%s/%s_%s' % (singan_opt.out, opts.singan_model, model_dir[-1]))
        # print(out_dir)       
        shutil.move(os.path.join(out_dir, '0.png'), '%s/%s_%s/%s.png' % (singan_opt.out, opts.singan_model, model_dir[-1], opts.input_picture[:-4]))
        shutil.rmtree('%s/RandomSamples' % singan_opt.out)

        del Gs
        del Zs
        del NoiseAmp
        del seg_imgs
        del functions
        del SinGAN_generate_V3

        
        torch.cuda.empty_cache()

        img = cv2.imread('%s/%s_%s/%s.png' % (singan_opt.out, opts.singan_model, model_dir[-1], opts.input_picture[:-4]))
        img = cv2.resize(img, (256,256))
        if opts.source_picture is not None:
            src = cv2.imread(opts.source_picture)
            src = cv2.resize(src, (opts.size,opts.size))
            print('PSNR: %f' % psnr(src, img))
            print('SSIM: %f' % compare_ssim(src, img,  multichannel=True))
        # cv2.imshow(model_dir, img)
        del singan_opt

    # cv2.waitKey(0)
if __name__ == '__main__':
    main()
