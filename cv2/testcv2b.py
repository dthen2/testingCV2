import cv2
import numpy as np

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# 輪郭を抽出するプログラムを作った
# 行列演算を使わずに for 文を回しているので、遅いったらありゃしない
# testcv2.py の動画バージョン

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
    y_size, x_size, c = img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)
    print(x_size_s, y_size_s)
    #imgs = cv2.resize(img, (x_size_s, y_size_s))
    #cv2.imshow(filen, imgs)
    #cv2.waitKey(1)

    sh = 6000.0
    #count = 0
    while True:
        ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        y_array = []
        for y in range(1, y_size_s-2):
            x_array = []
            for x in range(1, x_size_s-2):
                # ここで、[x,y,c]というインデックスの付け方は、np.arrayでしか使えず、普通のリストではNGなので注意。
                # float 等の変換は、配列に対しては使えないので注意
                dx =  (float(imgs[y,x+1,0]) + float(imgs[y-1,x+1,0]) + float(imgs[y+1,x+1,0]) \
                     - float(imgs[y,x,  0]) - float(imgs[y-1,x,  0]) - float(imgs[y+1,x,  0]))**2 # Blue
                dx += (float(imgs[y,x+1,1]) + float(imgs[y-1,x+1,1]) + float(imgs[y+1,x+1,1]) \
                     - float(imgs[y,x,  1]) - float(imgs[y-1,x,  1]) - float(imgs[y+1,x,  1]))**2 # Green
                dx += (float(imgs[y,x+1,2]) + float(imgs[y-1,x+1,2]) + float(imgs[y+1,x+1,2]) \
                     - float(imgs[y,x,  2]) - float(imgs[y-1,x,  2]) - float(imgs[y+1,x,  2]))**2 # Red
                dy =  (float(imgs[y+1,x,0]) + float(imgs[y+1,x+1,0]) + float(imgs[y+1,x-1,0]) \
                     - float(imgs[y,  x,0]) - float(imgs[y  ,x+1,0]) - float(imgs[y  ,x-1,0]))**2
                dy += (float(imgs[y+1,x,1]) + float(imgs[y+1,x+1,1]) + float(imgs[y+1,x-1,1]) \
                     - float(imgs[y,  x,1]) - float(imgs[y  ,x+1,1]) - float(imgs[y  ,x-1,1]))**2
                dy += (float(imgs[y+1,x,2]) + float(imgs[y+1,x+1,2]) + float(imgs[y+1,x-1,2]) \
                     - float(imgs[y,  x,2]) - float(imgs[y  ,x+1,2]) - float(imgs[y  ,x-1,2]))**2
                diff = dx+dy
                if diff > sh:
                    #x_array.append(np.array([250.0,250.0,250.0]))
                    x_array.append([250.0,250.0,250.0])
                else:
                    #x_array.append(np.array([0.0,0.0,0.0]))
                    x_array.append([0.0,0.0,0.0])
            #y_array.append(np.array(x_array))
            y_array.append(x_array)
        img_diffx = np.array(y_array)
        #print(type(img_diffx))
        cv2.imshow(filen, imgs)
        cv2.imshow('diff', img_diffx)

        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            break

    #cv2.waitKey(0)
    cv2.destroyAllWindows()
