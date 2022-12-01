import copy
import cv2
import numpy as np

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# カエルの目作ってみた。データ変換について学んだ所がミソ
# numpyへの関数適用について、maximum, abs が要素毎に使えることを学んだ。
# 画像データは 'uint8'：符号無し8ビットなので、ちょっと面倒

if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、照度がへぼい
    image = 'normal_cam'
    #image = 'file'
    video_file_name = 'mov_hts-samp009.mp4' # 飛行機
    video_file_name ='D0002160807_00000.mp4' # 街
    video_file_name ='D0002060316_00000.mp4' # クルマ

    if image == 'normal_cam':
        filen = 'normal_cam'
        cam = cv2.VideoCapture(0) # PCのカメラ
        blurred_rate = 1.0
        sh = 30
        amp = 5
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        cam = cv2.VideoCapture(1) # USBカメラ
        blurred_rate = 1.0
        sh = 10
        amp = 30
    elif image == 'file':
        filen = 'Video_file'
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
        blurred_rate = 2.0
        sh = 30
        amp = 5
    else:
        print('Please specify image')

    ret, img = cam.read() # ret, img = にしないと動かない
    print(img.shape)
    y_size, x_size, c = img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)
    print(x_size_s, y_size_s)
    #cv2.waitKey(1) # 何かキーを押すまで待つ
    img_last = cv2.resize(img, (x_size_s, y_size_s))

    ceiling_value = 250
    while True:
        ret, img = cam.read() # ret, img = にしないと動かない
        img_new = cv2.resize(img, (x_size_s, y_size_s))
        img_diff_v = abs(np.array(img_new,dtype='int16') - np.array(img_last,dtype='int16')) - sh
        img_diff_c = np.minimum(np.maximum(img_diff_v*amp, 0), ceiling_value)
        img_diff = np.array(img_diff_c,dtype='uint8')
        cv2.imshow(filen, img)
        cv2.imshow('diff', img_diff)
        img_last = copy.copy(img_new)
    
        key = cv2.waitKey(50) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            break

    #cv2.waitKey(0)
    cv2.destroyAllWindows()