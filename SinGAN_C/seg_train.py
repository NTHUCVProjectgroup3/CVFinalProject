from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
from skimage import io as img
from skimage.transform import resize
from skimage import color
from datasets.cityscapes import Cityscapes
from datasets.voc import VOCSegmentation
from torch.utils.data import DataLoader
import cv2
import torch

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--seg_dir', help='input reference dir', default='Input/Images')
    parser.add_argument('--mode', help='task to be done', default='seg_train')
    parser.add_argument('--test_seg', help='', default='aachen_seg.png')
    parser.add_argument('--dataset', choices=['voc', 'cityscapes'], help='', default='cityscapes')
    opt = parser.parse_args()
    opt.input_name = opt.dataset + 'xxxx'
    opt = functions.post_config(opt)
    opt.max_size = 256
    Gs = []
    Zs = []
    NoiseAmp = []
    segs = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        if opt.dataset == 'cityscapes':
            dataset = Cityscapes('../Datasets/cityscapes', 1024)
            real, seg = dataset[0]
        elif opt.dataset == 'voc':
            dataset = VOCSegmentation('../Datasets/VOCdevkit/VOC2012', 500)
            real, seg = dataset[0]

        real = torch.unsqueeze(real, 0)
        real = functions.adjust_scales2image(real, opt)

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        seg_train(opt,loader,Gs,Zs,segs,NoiseAmp)

        segs = torch.load('%s/segs.pth' % (opt.out_))
        
        test_seg = img.imread('%s/%s' % (opt.seg_dir, opt.test_seg))
        test_seg = cv2.resize(test_seg, (256, 256), interpolation=cv2.INTER_NEAREST)
        test_seg = functions.np2torch(test_seg, opt)
        test_segs = []

        for i, img in enumerate(segs):
            h, w = img.shape[2:]

            curr_seg = resize(test_seg.cpu().numpy(), (1, 3, h, w), order=0)
            test_segs.append(torch.from_numpy(curr_seg).type(torch.cuda.FloatTensor))
    
        out = SinGAN_generate_V3(Gs, Zs, segs, NoiseAmp, opt, None, num_samples=10)