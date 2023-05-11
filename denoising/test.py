import os
import random
import numpy as np
import tensorflow as tf
from model import u_net
import datetime
from skimage.io import imread
import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import bm3d

class MyDatasets_test(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size, image_path, sigma=15):
        self.batch_size = batch_size
        self.image_path = image_path
        self.sigma = sigma

    def __len__(self):
        return len(self.image_path) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        input_img_path = self.image_path[i : i + self.batch_size]

        for j, img_file in enumerate(input_img_path):
            img = np.expand_dims(imread(img_file) / 255., axis=0)
            b, h, w, c = img.shape
            h_crop = h % 16
            w_crop = w % 16
            img = img[:,:h - h_crop, :w - w_crop, :]
            # add gaussion noisy
            noisy_img = img + np.random.normal(0, self.sigma / 255., img.shape)
        return img, noisy_img

if __name__ == '__main__':
    data_path = '../datasets/test/denoising'
    # datasets = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']
    datasets = ['CBSD68']

    test_path = []
    for dataset in tqdm.tqdm(datasets):
        image_path_train = os.listdir(os.path.join(data_path,dataset))
        for path in image_path_train:
            test_path.append(os.path.join(data_path, dataset, path))

    ds_test = MyDatasets_test(batch_size=1,image_path=test_path)

    model = u_net()
    model.summary()
    model.load_weights('weights/model_1000.h5')

    gt_saved_path = r'saved_images/gt'
    noisy_saved_path = r'saved_images/noisy'
    pre_saved_path = r'saved_images/pre'

    n_image = ds_test.__len__()
    psnr_pre = ['pre_psnr']
    ssim_pre = ['pre_ssim']
    psnr_bm3d = ['bm3d_psnr']
    ssim_bm3d = ['bm3d_ssim']
    psnr_noisy = ['noisy_psnr']
    ssim_noisy = ['noisy_ssim']

    for i in tqdm.tqdm(range(n_image)):
        img, noisy_img = ds_test.__getitem__(i)
        # 将噪声图片输入模型，得到输出
        prediction = model(noisy_img)
        pre = (np.clip(prediction[0], 0, 1) * 255).astype(np.uint8)
        gt = (img[0] * 255).astype(np.uint8)
        noisy = (np.clip(noisy_img[0], 0, 1) * 255).astype(np.uint8)

        # BM3D算法
        bm3d_pre = bm3d.bm3d_rgb(noisy,sigma_psd=15).astype(np.uint8)

        # 计算psnr，ssim
        psnr_pre.append(PSNR(pre, gt))
        ssim_pre.append(SSIM(pre, gt, channel_axis=-1))
        psnr_bm3d.append(PSNR(bm3d_pre, gt))
        ssim_bm3d.append(SSIM(bm3d_pre, gt, channel_axis=-1))
        psnr_noisy.append(PSNR(noisy, gt))
        ssim_noisy.append(SSIM(noisy, gt, channel_axis=-1))

        # # 保存图片
        # Image.fromarray(gt).save(os.path.join(gt_saved_path, 'gt_{}.png'.format(i)))
        # Image.fromarray(noisy).save(os.path.join(noisy_saved_path, 'noisy_{}_psnr_{:.4f}_ssim_{:.4}.png'.format(i, psnr_noisy[-1], ssim_noisy[-1])))
        # Image.fromarray(pre).save(os.path.join(pre_saved_path, 'pre_{}_psnr_{:.4f}_ssim_{:.4}.png'.format(i, psnr_pre[-1], ssim_pre[-1])))

    # 保存日志文件
    test_logs = r'logs/test'
    np.savetxt(os.path.join(test_logs, 'test_psnr_ssim.csv'), [p for p in zip(psnr_noisy, ssim_noisy,psnr_pre,ssim_pre,psnr_bm3d,ssim_bm3d)],delimiter=',', fmt='%s')
    print('average predicted psnr: {:.4f}\naverage predicted ssim: {:.4f}\naverage bm3d psnr: {:.4f}\naverage bm3d ssim: {:.4f}\naverage noisy psnr: {:.4f}\naverage noisy ssim: {:.4f}\n'
          .format(np.mean(psnr_pre[1:]),np.mean(ssim_pre[1:]),np.mean(psnr_bm3d[1:]),np.mean(ssim_bm3d[1:]),np.mean(psnr_noisy[1:]),np.mean(ssim_noisy[1:])))
