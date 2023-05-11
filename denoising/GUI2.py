import tkinter
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from model import u_net
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.io import imread

# 设置图片保存路径
outfile = 'saved_images/GUI'
# 创建一个界面窗口
win = tkinter.Tk()
win.title("Image denoing")
win.geometry("1440x810")
# 打开图像,转为tkinter兼容的对象,
img = Image.open('saved_images/GUI/background1.jpg').resize([1440, 810])
img = ImageTk.PhotoImage(img)
# 创建画布，将图像作为画布背景, 铺满整个窗口
canvas = Canvas(win, width=1440, height=810)  # 设置画布的宽、高
canvas.place(x=0, y=0)
canvas.create_image(720, 405, image=img)

# 设置全局变量
# original = Image.new('RGB', (480, 320))
# save_img = Image.new('RGB', (480, 320))
# count = 0
img1 = tkinter.Label(win)
img2 = tkinter.Label(win)
img3 = tkinter.Label(win)

# 实现在本地电脑选择图片
def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    e.set(select_file)
    load = Image.open(select_file)
    base_width = 320
    w_percent = base_width / float(load.size[0])
    base_hight = int(float(load.size[1]) * float(w_percent))
    if base_hight % 16 != 0:
        base_hight = base_hight - (base_hight % 16)

    load = load.resize((base_width,base_hight))

    # 声明全局变量
    global original
    original = load

    render = ImageTk.PhotoImage(original)

    label1 = tkinter.Label(win, text="原始图片",font=('宋体', 20),fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label1.place(x=200, y=120)

    global img1
    img1.destroy()
    img1 = tkinter.Label(win, image=render)
    img1.image = render
    img1.place(x=100, y=180)

def add_noise():
    temp = original
    temp_img = np.array(temp)
    noisy_img = np.clip(temp_img + np.random.normal(0, 15, temp_img.shape),0,255).astype('uint8')


    label_n = tkinter.Label(win, text="噪声图片",font=('宋体', 20),fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label_n.place(x=660, y=120)

    new_img = Image.fromarray(noisy_img)
    render = ImageTk.PhotoImage(new_img)
    global img3
    img3.destroy()
    img3 = tkinter.Label(win, image=render)
    img3.image = render
    img3.place(x=550, y=180)

    psnr_noisy = (PSNR(noisy_img, temp_img))
    ssim_noisy = (SSIM(noisy_img, temp_img, channel_axis=-1))

    label_1 = tkinter.Label(win, text="PSNR={:.4f},SSIM={:.4f}".format(psnr_noisy,ssim_noisy),font=('Times New Roman', 10),fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label_1.place(x=640, y=160)


def model_test(img_file, sigma=15):
    model = u_net()
    model.load_weights('weights/model_1000.h5')
    img = np.expand_dims(img_file / 255., axis=0)
    noisy_img = img + np.random.normal(0, sigma / 255., img.shape)
    prediction = model(noisy_img)
    pre = (np.clip(prediction[0], 0, 1) * 255).astype(np.uint8)
    gt = (img[0] * 255).astype(np.uint8)
    noisy = (np.clip(noisy_img[0], 0, 1) * 255).astype(np.uint8)

    psnr_pre = (PSNR(pre, gt))
    ssim_pre = (SSIM(pre, gt, channel_axis=-1))
    psnr_noisy = (PSNR(noisy, gt))
    ssim_noisy = (SSIM(noisy, gt, channel_axis=-1))

    label_2 = tkinter.Label(win, text="PSNR={:.4f},SSIM={:.4f}".format(psnr_pre,ssim_pre),font=('Times New Roman', 10),fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label_2.place(x=1090, y=160)

    return noisy, pre

def denoising():
    temp = original
    temp_img = np.array(temp)
    noisy, pre = model_test(temp_img)
    #print(f'{temp.size}---->{new_im.size}')
    #render = ImageTk.PhotoImage(tensor_to_PIL(new_img))
    new_img = Image.fromarray(pre)
    render = ImageTk.PhotoImage(new_img)

    label2 = tkinter.Label(win, text="修复图片",font=('宋体', 20), fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label2.place(x=1100, y=120)

    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=1000, y=180)
    global save_img
    save_img = new_img

# 保存函数
def save():
    save_img.save(os.path.join(outfile,'pre.png'))

if __name__ == '__main__':
    # 显示路径
    e = tkinter.StringVar()
    e_entry = tkinter.Entry(win, width=68, textvariable=e)
    e_entry.pack()

    # 设置选择图片的按钮
    button1 = tkinter.Button(win, text="选择图片", command=choose_file,font=('宋体', 20),
                             fg='black',bg='SkyBlue', activebackground='black',activeforeground='white')
    button1.pack()

    # 设置增加噪声的按钮
    button3 = tkinter.Button(win, text="增加噪声", command=add_noise,font=('宋体', 20), fg='black',
                             bg='Cyan', activebackground='black',activeforeground='white')
    button3.place(x=150, y=700)

    # 设置去噪声的按钮
    button3 = tkinter.Button(win, text="去噪", command=denoising,font=('宋体', 20), fg='black',
                             bg='Cyan', activebackground='black',activeforeground='white')
    button3.place(x=500, y=700)

    # 设置保存图片的按钮
    button2 = tkinter.Button(win, text="保存图片", command=save,font=('宋体', 20), fg='black',
                             bg='Cyan', activebackground='black',activeforeground='white')
    button2.place(x=800, y=700)


    # 设置退出按钮
    button0 = tkinter.Button(win, text="退出", command=win.quit,font=('宋体', 20), fg='black',
                             bg='IndianRed', activebackground='black',activeforeground='white')
    button0.place(x=1200, y=700)
    win.mainloop()