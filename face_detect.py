import cv2
import logging
import os

#设置日志
logging.basicConfig(level = logging.INFO, 
                    format = '%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#待检测图片路径
images_list_path = os.listdir('images')
print(images_list_path)

#读取图片
images = []
gray_images = []
for _image in images_list_path:
    logger.info('读取图片： ' + _image)
    _image_path = os.path.join('./images/', _image)
    image = cv2.imread(_image_path)
    images.append(image)
    #将图片转换为灰度图
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray_images.append(gray_image)
logger.info('图片读取完成')

#显示图片
# image = cv2.imread('E:/jupyter/face_detect/images/1.jpg')
# cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('input_image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#检测人脸
logger.info('人脸检测...')
#调用训练好的人脸参数数据，进行人脸检测
face_cascade = cv2.CascadeClassifier(r'./human_front_face.xml')
for i in range(len(images)):
    faces = face_cascade.detectMultiScale(gray_images[i]) #scaleFactor = 1.1, minNeighbors = 3, minSize = (3,3)

    search_info = "检测到 %d 张人脸." % len(faces)
    logger.info(search_info)

    #绘制人脸的矩阵区域
    for (x, y, w, h) in faces:
        cv2.rectangle(images[i], (x,y), (x+w, y+h), (0,0,255), 2)

    #显示图片
    cv2.imshow('Find faces! ', images[i])
    cv2.waitKey(500)