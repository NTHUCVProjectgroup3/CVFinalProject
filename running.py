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

from skimage.transform import resize

def transfer_to_color_map(label_img, opts):
    if opts.dataset == 'cityscapes':
        return Cityscapes.decode_target(label_img)
    elif opts.dataset == 'voc':
        return VOCSegmentation.decode_target(label_img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_picture', type=str, help='select one image from your disk', required=True)
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'voc'], help='select one dataset for trained model', default='cityscapes')
    parser.add_argument('--deeplabv3plus_path', type=str, help='select deeplabv3plus_mobilenet pretrained model path', required=True)
    parser.add_argument('--pix2pix_path', type=str, help='select pix2pix pretrained model path', default=None)
    parser.add_argument('--singan_model', type=str, help='select one trained singan model', default=None)
    parser.add_argument('--mode', type=str, default='A', choices=['A', 'B'],  help='')

    parser.add_argument('--singan_mode', type=str, help='', default='random_samples')
    # parser.add_argument('--singan_ref_mode_name', type=str, help='', default=None)

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

    if opts.dataset == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset == 'voc':
        opts.num_classes = 21

    out_name = opts.input_picture[:-4]
    deeplabv3plus_mobilenet = network.deeplabv3plus_mobilenet(num_classes=opts.num_classes, output_stride=16)
    deeplab_state = torch.load(opts.deeplabv3plus_path)
    deeplabv3plus_mobilenet.load_state_dict(deeplab_state["model_state"])
    deeplabv3plus_mobilenet.eval()

    deeplabv3plus_mobilenet.cuda()

    img = Image.open(opts.input_picture).convert('RGB')
    val_transform = transforms.Compose([
        # et.ExtResize( 512 ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_img = val_transform(img).cuda()
    tensor_img = torch.unsqueeze(tensor_img, 0)

    with torch.no_grad():
        pred = deeplabv3plus_mobilenet(tensor_img)
        labels = pred.detach().max(dim=1)[1].cpu().numpy()

        output = transfer_to_color_map(labels, opts).astype(np.uint8)
        output = output.squeeze()

        seg_img = Image.fromarray(output)
        seg_img.save('%s_seg.png' % out_name)
        overlay = (np.array(img, dtype=np.float) * 0.65 + output.astype(np.float) * 0.35).astype(np.uint8)
        Image.fromarray(overlay).save('%s_overlay.png' % out_name)

    seg_img = seg_img.resize((256, 256), Image.NEAREST)

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
            tensor_img2 = transform2(seg_img)
            tensor_img2 = torch.unsqueeze(tensor_img2, 0)
            pix2pix_model.real_A = tensor_img2
            pix2pix_model.forward()

            fake = pix2pix_model.fake_B.data
            fake = fake[0].cpu().float().numpy()

            fake = (np.transpose(fake, (1, 2, 0)) + 1) / 2.0 * 255.0
            fake = fake.astype(np.uint8)

        fake_img = Image.fromarray(fake)
        fake_img.save('%s_fake.png' % out_name)

        if opts.singan_model is not None:

            singan_opt = Namespace(min_size=25, max_size=256, scale_factor=0.75, nc_im=3, not_cuda=False,
                                   mode=opts.singan_mode, alpha=10, gen_start_scale=0, manualSeed=None,
                                   input_name=opts.input_picture, out='SinGAN_Output', ker_size=3, num_layer=5, nc_z=3,
                                   niter=2000, noise_amp=0.1, nfc=32, min_nfc=32)
            singan_opt = functions.post_config(singan_opt)

            singan_dir = 'SinGAN/TrainedModels/%s/scale_factor=%f,alpha=%d' % (
                          opts.singan_model, singan_opt.scale_factor, singan_opt.alpha)
            originals = torch.load(os.path.join(singan_dir, 'reals.pth'))
            h, w = originals[-1].shape[2:]
            fake_imgs = []

            for i, img in enumerate(originals):
                h, w = img.shape[2:]

                fake_tmp = fake_img.resize((w, h))
                fake = np.array(fake_tmp)
                fake = fake.transpose(2, 0, 1)
                fake = np.expand_dims(fake, axis=0).astype('float')
                fake = fake / 255.0

                fake_tensor = torch.from_numpy(fake).type(torch.cuda.FloatTensor)
                fake_tensor = (fake_tensor - 0.5) * 2
                fake_tensor = fake_tensor.clamp(-1, 1)

                fake_imgs.append(fake_tensor)

            if not os.path.exists(singan_opt.out):
                os.mkdir(singan_opt.out)

            # functions.adjust_scales2image(fake_tensor, singan_opt)

            Gs = torch.load(os.path.join(singan_dir, 'Gs.pth'))
            Zs = torch.load(os.path.join(singan_dir, 'Zs.pth'))

            # reals = functions.creat_reals_pyramid(fake_tensor,fake_imgs,singan_opt)
            # reals = torch.load(os.path.join(singan_dir, 'reals.pth'))
            NoiseAmp = torch.load(os.path.join(singan_dir, 'NoiseAmp.pth'))
            in_s = functions.generate_in2coarsest(fake_imgs, 1, 1, singan_opt)
            SinGAN_generate(Gs, Zs, fake_imgs, NoiseAmp, singan_opt, in_s=in_s, gen_start_scale=0)

    elif opts.mode == 'B':
        if opts.singan_model is None:
            raise Exception('Error: mode B should specify singan model!')

        singan_opt = Namespace(min_size=25, max_size=256, scale_factor=0.7722260524731895, nc_im=3, not_cuda=False,
                               mode=opts.singan_mode, alpha=10, gen_start_scale=0, manualSeed=None,
                               input_name=opts.input_picture, out='SinGAN_Output', ker_size=3, num_layer=5, nc_z=3,
                               niter=2000, noise_amp=0.1, nfc=32, min_nfc=32)
        singan_opt = functions.post_config(singan_opt)

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


if __name__ == '__main__':
    main()
