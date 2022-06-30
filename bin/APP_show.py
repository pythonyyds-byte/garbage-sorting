import tkinter as tk
from tkinter import ttk
import win32ui
from PIL import Image,ImageTk
import cv2 as cv
import time
import app_test
#打开文件的选择对话框
def Open_img():
    dlg = win32ui.CreateFileDialog(1)  # 1表示打开文件对话框
    dlg.SetOFNInitialDir('D:/Python')  # 设置打开文件对话框中的初始显示目录
    dlg.DoModal()
    filepath = dlg.GetPathName()  # 获取选择的文件名称
    return filepath

def Img_choose():
    filepath = Open_img()  #选择图片
    Img_predict(filepath)
    Img_show(filepath)

def Img_predict(filepath):

    result = app_test.predict(filepath)
    text1.delete('1.0','end')
    for i in result:
        text1.insert('insert','概率: {:.3f}, 预测垃圾名称: {}\n'.format(i['c'],i['name']))  #图形化界面显示预测结果



def Img_show(filepath):
    global image
    #图形化界面显示图片
    img = Image.open(filepath)
    img = img.resize((400,400))
    image = ImageTk.PhotoImage(img)
    label2 = ttk.Label(text='图片',image=image)
    label2.place(x=20,y=100)

#拍照获取图片
def get_camera():
    global image1
    path = r'../dataset_camera/'  # 图像保存路径
    images = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    filepath = path + images + ".jpg"

    cap = cv.VideoCapture(0)# 调用摄像头
    flag = cap.isOpened()
    while (flag):
        ret, frame = cap.read()
        cv.imshow("Capture_Paizhao", frame)
        k = cv.waitKey(1) & 0xFF
        if k == ord('s'):  # 按下s键，进入下面的保存图片操作
            cv.imwrite(filepath, frame)
            break
        else:
            continue
    cap.release()
    cv.destroyWindow("Capture_Paizhao")

    Img_show(filepath)
    Img_predict(filepath)


#创界面
win = tk.Tk()
win.title("垃圾分类")
win.geometry("450x700")

bt1 = tk.Button(win,text='选择图片',command=Img_choose,width=10)
bt1.place(x=100,y=600)
bt1 = tk.Button(win,text='拍照',width=10,command=get_camera)
bt1.place(x=260,y=600)
text1 = tk.Text(win,height=3,width=55)
text1.place(x=30,y=35)
label1 = tk.Label(win,text= '预测结果')
label1.place(x=30,y=10)

win.mainloop()




