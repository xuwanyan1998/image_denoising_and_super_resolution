import os
# from data import DIV2K
from model import edsr
from train import EdsrTrainer
from model import resolve_single,load_image
import numpy as np
from PIL import Image
from imresize import imresize
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

weights_file = 'weights/edsr-16-x4/weights.h5'
model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)


def test_dataset(dataset_path):
    psnr_total = 0
    ssim_total = 0
    bicubic_psnr_total = 0
    bicubic_ssim_total = 0
    idx = 0
    dataset_path = os.path.join(dataset_path, 'HR')
    for img in os.listdir(dataset_path):
        hr = load_image(os.path.join(dataset_path,img))
        if len(hr.shape) <3:
            hr = hr[..., np.newaxis]
            hr = np.concatenate([hr] * 3, 2)

        width_pad = hr.shape[0] % scale
        highth_pad = hr.shape[1] % scale

        if width_pad != 0:
            hr_pad = np.pad(hr,((0,scale - width_pad),(0,0),(0,0)),'edge')
        else:
            hr_pad = hr

        if highth_pad != 0:
            hr_pad = np.pad(hr_pad, ((0, 0), (0,scale - highth_pad) ,(0,0)), 'edge')
        else:
            hr_pad = hr_pad

        lr = imresize(hr_pad, 0.25, method='bicubic')
        # bicubic 算法
        bicubic_sr = imresize(lr, 4, method='bicubic')
        sr = resolve_single(model, lr)
        sr = np.array(sr).astype(np.uint8)
        if width_pad != 0:
            sr = sr[:width_pad - scale, :, :]
            bicubic_sr = bicubic_sr[:width_pad - scale, :, :]
        if highth_pad != 0:
            sr = sr[:, :highth_pad - scale, :]
            bicubic_sr = bicubic_sr[:, :highth_pad - scale, :]
        psnr = PSNR(sr, hr)
        ssim = SSIM(sr, hr, channel_axis=-1)
        bicubic_psnr = PSNR(bicubic_sr, hr)
        bicubic_ssim = SSIM(bicubic_sr, hr, channel_axis=-1)

        print(img,psnr,ssim,bicubic_psnr,bicubic_ssim)
        psnr_total += psnr
        ssim_total += ssim
        bicubic_psnr_total += bicubic_psnr
        bicubic_ssim_total += bicubic_ssim
        idx += 1
    print('mean PSNR: {:.4f}, mean SSIM: {:.4f}'.format(psnr_total/idx, ssim_total/idx))
    print('mean bicubic PSNR: {:.4f}, mean bicubic SSIM: {:.4f}'.format(bicubic_psnr_total / idx, bicubic_ssim_total / idx))

def test_x4(dataset_path):
    psnr_total = 0
    ssim_total = 0
    idx = 0
    for img in os.listdir(os.path.join(dataset_path,'LR_bicubic', 'x4')):
        lr = load_image(os.path.join(dataset_path,'LR_bicubic', 'x4',img))
        sr = resolve_single(model, lr)
        sr = np.array(sr).astype(np.uint8)
        hr = load_image(os.path.join(dataset_path, 'HR', img.replace('x4.png','.png')))[:sr.shape[0],:sr.shape[1]]
        psnr = PSNR(sr, hr)
        ssim = SSIM(sr, hr, channel_axis=-1)
        print(img, psnr, ssim)
        psnr_total += psnr
        ssim_total += ssim
        idx += 1
    print('mean PSNR: {:.4f}, mean SSIM: {:.4f}'.format(psnr_total / idx, ssim_total / idx))


if __name__ == '__main__':
    dataset_path = '../datasets/test/super_resolution/Set5'
    test_dataset(dataset_path)
