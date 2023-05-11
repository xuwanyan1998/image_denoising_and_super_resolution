import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from model import u_net
import datetime
from skimage.io import imread
import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


def train_step(model, de_image, gt_image, steps):
    with tf.GradientTape() as tape:
        predictions = model(de_image, training=True)
        # loss = loss_func(gt_image, predictions)
        loss = tf.reduce_mean(tf.abs(predictions - gt_image))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    psnr = PSNR(gt_image, predictions.numpy())
    steps = tf.cast(steps, dtype=tf.int64)
    with summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=steps)
        tf.summary.scalar('train_psnr', psnr, step=steps)
    return loss, psnr

def valid_step(model, de_image, gt_image, steps):
    predictions = model(de_image)
    batch_loss = tf.reduce_mean(tf.abs(predictions - gt_image))
    psnr = PSNR(gt_image, predictions.numpy())
    steps = tf.cast(steps, dtype=tf.int64)
    with summary_writer.as_default():
        tf.summary.scalar('val_loss', batch_loss, step=steps)
        tf.summary.scalar('val_psnr', psnr, step=steps)
    return batch_loss, psnr

def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        epoch_train_loss = 0
        epoch_train_psnr = 0
        epoch_val_loss = 0
        epoch_val_psnr = 0
        train_steps_per_epoch = ds_train.__len__()
        valid_steps_per_epoch = ds_valid.__len__()

        for step in tqdm.tqdm(range(train_steps_per_epoch)):
            patch_img, patch_noisy_img = ds_train.__getitem__(step)
            train_steps = step + (epoch - 1)*train_steps_per_epoch
            train_loss, train_psnr = train_step(model, patch_noisy_img, patch_img, train_steps)
            epoch_train_loss += train_loss
            epoch_train_psnr += train_psnr

        for step in tqdm.tqdm(range(valid_steps_per_epoch)):
            patch_img, patch_noisy_img = ds_valid.__getitem__(step)
            val_steps = step + (epoch - 1) * valid_steps_per_epoch
            val_loss, val_psnr = valid_step(model, patch_noisy_img, patch_img, val_steps)
            epoch_val_loss += val_loss
            epoch_val_psnr += val_psnr

        epoch_train_loss = epoch_train_loss / len(ds_train)
        epoch_train_psnr = epoch_train_psnr / len(ds_train)
        epoch_val_loss = epoch_val_loss / len(ds_valid)
        epoch_val_psnr = epoch_val_psnr / len(ds_valid)

        epoch = tf.cast(epoch,dtype=tf.int64)
        with summary_writer.as_default():
            tf.summary.scalar('epoch_train_loss', epoch_train_loss, step=epoch)
            tf.summary.scalar('epoch_train_psnr', epoch_train_psnr, step=epoch)
            tf.summary.scalar('epoch_val_loss', epoch_val_loss, step=epoch)
            tf.summary.scalar('epoch_val_psnr', epoch_val_psnr, step=epoch)

        # 每个epoch打印相应信息
        logs = 'Epoch={},Loss:{:.4f},psnr:{:.4f},Valid Loss:{:.4f},Valid psnr:{:.4f}'
        print(logs.format(epoch, epoch_train_loss, epoch_train_psnr, epoch_val_loss, epoch_val_psnr))
        # save model
        if epoch % 10 == 0:
            model.save_weights('model/model_{}.h5'.format(epoch))
            print('.................save model successfully!...............')

class MyDatasets(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, image_path, sigma=15, patch_size=128, randomfliping=False):
        self.batch_size = batch_size
        self.image_path = image_path
        self.sigma = sigma
        self.patch_size = patch_size
        self.randomfliping = randomfliping

    def __len__(self):
        return len(self.image_path) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.image_path[i : i + self.batch_size]
        patch_img = np.zeros((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype="float32")
        patch_noisy_img = np.zeros((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = imread(path) / 255.
            H, W, C = img.shape
            # random crop to patch_size
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_img[j] = img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # add gaussion noisy
            patch_noisy_img[j] = patch_img[j] + np.random.normal(0, self.sigma / 255., patch_img[j].shape)

        if self.randomfliping:
            if random.randint(1, 2) == 1:  # random flip
                patch_img = np.flip(patch_img, axis=1)
                patch_noisy_img = np.flip(patch_noisy_img, axis=1)
            if random.randint(1, 2) == 1:
                patch_img = np.flip(patch_img, axis=2)
                patch_noisy_img = np.flip(patch_noisy_img, axis=2)

        return patch_img, patch_noisy_img

class MyDatasets2(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, image_files, sigma=15, patch_size=128, randomfliping=False):
        self.batch_size = batch_size
        self.image_files = image_files
        self.sigma = sigma
        self.patch_size = patch_size
        self.randomfliping = randomfliping

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_files = self.image_files[i : i + self.batch_size]
        patch_img = np.zeros((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype="float32")
        patch_noisy_img = np.zeros((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype="float32")
        for j, img_file in enumerate(batch_input_img_files):
            H, W, C = img_file.shape
            # random crop to patch_size
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_img[j] = img_file[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # add gaussion noisy
            patch_noisy_img[j] = patch_img[j] + np.random.normal(0, self.sigma / 255., patch_img[j].shape)

        if self.randomfliping:
            if random.randint(1, 2) == 1:  # random flip
                patch_img = np.flip(patch_img, axis=1)
                patch_noisy_img = np.flip(patch_noisy_img, axis=1)
            if random.randint(1, 2) == 1:
                patch_img = np.flip(patch_img, axis=2)
                patch_noisy_img = np.flip(patch_noisy_img, axis=2)

        return patch_img, patch_noisy_img

if __name__ == '__main__':
    # 加载和预处理数据集
    data_path = '../datasets/train'
    # datasets = ['BSD400','DIV2K','WaterlooED']
    datasets = ['BSD400']

    train_path = []
    val_path = []
    for dataset in datasets:
        image_path_train = os.listdir(os.path.join(data_path,dataset))
        for i, path in enumerate(image_path_train):
            if i % 10 == 9:
                val_path.append(os.path.join(data_path, dataset, path))
            else:
                train_path.append(os.path.join(data_path, dataset, path))

    ## load image files into RAM
    ## 将数据加载到内存里，三个数据集一起训练要求内存大于80G，训练速度块
    # train_images = []
    # val_images = []
    # print('...............load train data.................')
    # for path in tqdm.tqdm(train_path):
    #     train_images.append(imread(path) / 255.)
    #
    # print('...............load valid data.................')
    # for path in tqdm.tqdm(val_path):
    #     val_images.append(imread(path) / 255.)
    #
    # ds_train = MyDatasets2(batch_size=16,image_files=train_images)
    # ds_valid = MyDatasets2(batch_size=16,image_files=val_images)

    ## 不将数据提前加载到内存，训练速度慢，但是对内存大小要求低
    ds_train = MyDatasets(batch_size=16,image_path=train_path)
    ds_valid = MyDatasets(batch_size=16,image_path=val_path)
    epochs = 100

    # 构建模型
    model = u_net(height=128, width=128)
    model.summary()
    # model.load_weights('model/model_30.h5')

    optimizer = optimizers.adam_v2.Adam()

    # 训练数据记录
    log_dir = "logs/train/"
    summary_writer = tf.summary.create_file_writer(log_dir + datetime.datetime.now().strftime("%Y%m%d-%H"))
    # 训练
    train_model(model, ds_train, ds_valid, epochs)
