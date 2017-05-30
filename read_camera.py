# -*- coding:utf-8 -*-
__author__ = '万壑'

import cv2
from train_model import Model
from read_data import read_name_list

class Camera_reader(object):
    #在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128


    def build_camera(self):
        #opencv文件中人脸级联文件的位置，用于帮助识别图像或者视频流中的人脸
        face_cascade = cv2.CascadeClassifier('E:\openCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
        #读取dataset数据集下的子文件夹名称
        name_list = read_name_list('D:\myProject\pictures\dataset')

        #打开摄像头并开始读取画面
        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()

        while success and cv2.waitKey(1) == -1:
             success, frame = cameraCapture.read()
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #图像灰化
             faces = face_cascade.detectMultiScale(gray, 1.3, 5) #识别人脸
             for (x, y, w, h) in faces:
                 ROI = gray[x:x + w, y:y + h]
                 ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                 label,prob = self.model.predict(ROI)  #利用模型对cv2识别出的人脸进行比对
                 if prob >0.7:    #如果模型认为概率高于70%则显示为模型中已有的label
                     show_name = name_list[label]
                 else:
                     show_name = 'Stranger'
                 cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                 frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  #在人脸区域画一个正方形出来
             cv2.imshow("Camera", frame)

        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = Camera_reader()
    camera.build_camera()


