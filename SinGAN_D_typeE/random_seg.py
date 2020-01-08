from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from skimage.transform import resize

from datasets.cityscapes import Cityscapes
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples', default='random_samples')
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if False:
        pass
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        if opt.mode == 'random_samples':
            real = functions.read_image(opt)
            singan_dir = './TrainedModels/%s/scale_factor=%f_seg' % (
                'cityscapes_010620_1709', opt.scale_factor)

            originals = torch.load(os.path.join(singan_dir, 'segs.pth'))
            segs = []

            seg_img = Image.open('%s/%s' % (opt.input_dir, opt.input_name)).convert('RGB')

            seg_img = transforms.Resize((256,256))(seg_img)
            #seg_img = transforms.RandomHorizontalFlip(1)(seg_img)
            seg_img = np.array(seg_img)
            seg_img = functions.np2torch(seg_img, opt)

            for i, img in enumerate(originals):
                h, w = img.shape[2:]

                curr_seg = resize(seg_img.cpu().numpy(), (1, 3, h, w), order=0)
                segs.append(torch.from_numpy(curr_seg).type(torch.cuda.FloatTensor))

            del originals
            
            Gs = torch.load(os.path.join(singan_dir, 'Gs.pth'))
            Zs = torch.load(os.path.join(singan_dir, 'Zs.pth'))
            NoiseAmp = torch.load(os.path.join(singan_dir, 'NoiseAmp.pth'))
            in_s = functions.generate_in2coarsest(segs,1,1,opt)

            opt.scale_factor = 0.7722260524731895
            SinGAN_generate_V3(Gs, Zs, segs, NoiseAmp, opt, num_samples=10)
