from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import os


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes | random_samples_fix_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    # for random_samples_fix_sizes
    parser.add_argument('--horizon_size', type=float, help='horizontal size for random samples', default=256)
    parser.add_argument('--vertical_size', type=float, help='vertical size for random samples', default=256)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
        
    
    if(os.path.exists(dir2save)):
        i = 2
        while 1:
            tmp_dir2save = dir2save + '_%d' % i
            if not os.path.exists(tmp_dir2save):
                dir2save = tmp_dir2save
                break
            i += 1
    if True:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        if opt.mode == 'random_samples':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,1,1,opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale, dir2save=dir2save)

        elif opt.mode == 'random_samples_arbitrary_sizes':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, dir2save=dir2save)
            
        elif opt.mode == 'random_samples_fix_sizes':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.scale_v = float(opt.vertical_size) / float(reals[-1].shape[2])
            opt.scale_h = float(opt.horizon_size) / float(reals[-1].shape[3])
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h, dir2save=dir2save)





