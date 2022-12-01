import copy
import cv2
import numpy as np
#import tensorflow as tf
#from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

#
# https://axa.biopapyrus.jp/ia/opencv/detect-contours.html
# https://qiita.com/rareshana/items/6a2f5e7396f28f6eee49

if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、2回目のrunで照度が下がる？？？
    image = 'normal_cam'
    #image = 'file'
    #image = 'picture'
    video_file_name = 'planes.mp4' # 飛行機
    video_file_name ='town.mp4' # 街
    video_file_name ='cars.mp4' # クルマ
    video_file_name ='IMG_2712.mov' # ナナ
    pict_file_name = 'nana.jpg'

    if image == 'normal_cam':
        filen = 'normal_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(0) # PCのカメラ
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(1) # USBカメラ
    elif image == 'file':
        filen = 'Video_file'
        frame_rate = 24.0
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
    elif image == 'picture': # 静止画
        # https://peaceandhilightandpython.hatenablog.com/entry/2015/12/23/214840
        filen = 'picture_file'
        img = cv2.imread(pict_file_name) # .画像ファイル
    else:
        print('Please specify image')
    
    blurred_rate = 3.0

    if image != 'picture':
        ret, img = cam.read() # ret, img = にしないと動かない
        y_size, x_size, c = img.shape
        print(x_size, y_size)
        x_size_s = int(x_size/blurred_rate)
        y_size_s = int(y_size/blurred_rate)
        # 動画保存準備
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('record_video.mp4', fmt, frame_rate, (x_size_s, y_size_s))
    else:
        y_size, x_size, c = img.shape
        print(x_size, y_size)
        x_size_s = int(x_size/blurred_rate)
        y_size_s = int(y_size/blurred_rate)

    imgs = cv2.resize(img, (x_size_s, y_size_s))

    repeat = True
    while repeat:
        if image != 'picture':
            ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_HSV = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)
        cv2.imshow('img_HSV', img_HSV)# ネがみたいの
        img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)
        cv2.imshow('img_HSV2', img_HSV)# それをぼかしたもの

        # detect tulips
        img_H, img_S, img_V = cv2.split(img_HSV)
        cv2.imshow('img_H', img_H)# 白黒
        cv2.imshow('img_S', img_S)# 白黒
        cv2.imshow('img_V', img_V)# 元画像のぼけた白黒
        _thre, img_flowers = cv2.threshold(img_H, 140, 255, cv2.THRESH_BINARY)
        cv2.imshow('img_flowers', img_flowers) #何か見つけた
        cv2.imwrite('tulips_mask.jpg', img_flowers)

        # find tulips
        contours, hierarchy = cv2.findContours(img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):
            print('Number', i)
            if len(contours[i]) > 0:

                # remove small objects
                if cv2.contourArea(contours[i]) < 500:
                    continue

                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                cv2.rectangle(imgs, (x, y), (x + w, y + h), (0, 0, 0), 10)
        cv2.imshow(filen, imgs)# 元の画像にバウンディングボックス

        # save
        if image == 'picture':
            cv2.imwrite('tulips_boundingbox.jpg', imgs)
            repeat = False
        else: # 動画
            cv2.imshow('diff_time', imgs)
            writer.write(imgs) # 画像を1フレーム分として書き込み

        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            break
    if image == 'picture':
        cv2.waitKey(0)
    else:
        writer.release() # ファイルを閉じる
        cam.release()
    cv2.destroyAllWindows()
