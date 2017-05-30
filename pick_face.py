#-*-coding:utf8-*-
__author__ = '万壑'

import os
import cv2
import time
from read_img import readAllImg

#从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def readPicSaveFace(sourcePath,objectPath,*suffix):
    try:
        #读取照片,注意第一个元素是文件名
        resultArray=readAllImg(sourcePath,*suffix)

        #对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下

        count = 1
        face_cascade = cv2.CascadeClassifier('E:\openCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i) != str:

              gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(gray, 1.3, 5)
              for (x, y, w, h) in faces:

                listStr = [str(int(time.time())), str(count)]  #以时间戳和读取的排序作为文件名称
                fileName = ''.join(listStr)

                f = cv2.resize(gray[y:(y + h), x:(x + w)], (200, 200))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count += 1


    except IOError:
        print "Error"

    else:
        print 'Already read '+str(count-1)+' Faces to Destination '+objectPath

if __name__ == '__main__':
     readPicSaveFace('D:\myProject\pictures\source-jerry','D:\myProject\pictures\picTest','.jpg','.JPG','png','PNG')











