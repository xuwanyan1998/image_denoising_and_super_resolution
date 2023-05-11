import tkinter
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from model import edsr
from model import resolve_single

# 设置图片保存路径
outfile = 'saved_images/GUI'
# 创建一个界面窗口
win = tkinter.Tk()
win.title("Super_resolution")
win.geometry("1440x810")
# 打开图像,转为tkinter兼容的对象,
img = Image.open('saved_images/GUI/background1.jpg').resize([1440, 810])
img = ImageTk.PhotoImage(img)
# 创建画布，将图像作为画布背景, 铺满整个窗口
canvas = Canvas(win, width=1440, height=810)  # 设置画布的宽、高
canvas.place(x=0, y=0)
canvas.create_image(720, 405, image=img)

# 设置全局变量
img1 = tkinter.Label(win)
img2 = tkinter.Label(win)

# 实现在本地电脑选择图片
def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    e.set(select_file)
    load = Image.open(select_file)

    # 声明全局变量
    global original
    original = load

    base_width = 300
    w_percent = base_width / float(load.size[0])
    base_hight = int(float(load.size[1]) * float(w_percent))

    load = load.resize((base_width,base_hight), resample=Image.NEAREST)

    render = ImageTk.PhotoImage(load)

    label1 = tkinter.Label(win, text="原始图片",font=('宋体', 20),fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label1.place(x=300, y=120)
    global img1
    img1.destroy()
    img1 = tkinter.Label(win, image=render)
    img1.image = render
    img1.place(x=200, y=160)

def model_test(img_file):
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights('weights/edsr-16-x4/weights.h5')
    sr = resolve_single(model, img_file)
    return sr

def super_resolution():
    temp = original
    temp_img = np.array(temp)
    sr = model_test(temp_img)

    original_pre = Image.fromarray(np.array(sr))

    base_width = 300
    w_percent = base_width / float(original_pre.size[0])
    base_hight = int(float(original_pre.size[1]) * float(w_percent))

    new_img = original_pre.resize((base_width,base_hight), resample=Image.BOX)

    render = ImageTk.PhotoImage(new_img)

    label2 = tkinter.Label(win, text="修复图片",font=('宋体', 20), fg='black',
                           bg='LightSkyBlue', activebackground='black',activeforeground='white')
    label2.place(x=1000, y=120)

    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=900, y=160)
    global save_img
    save_img = original_pre

# 保存函数
def save():
    global count
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

    # 设置保存图片的按钮
    button2 = tkinter.Button(win, text="保存图片", command=save,font=('宋体', 20), fg='black',
                             bg='Cyan', activebackground='black',activeforeground='white')
    button2.place(x=660, y=700)

    # 设置超分辨率的按钮
    button3 = tkinter.Button(win, text="超分辨率", command=super_resolution,font=('宋体', 20), fg='black',
                             bg='Cyan', activebackground='black',activeforeground='white')
    button3.place(x=150, y=700)

    # 设置退出按钮
    button0 = tkinter.Button(win, text="退出", command=win.quit,font=('宋体', 20), fg='black',
                             bg='IndianRed', activebackground='black',activeforeground='white')
    button0.place(x=1100, y=700)
    win.mainloop()