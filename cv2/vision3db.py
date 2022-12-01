from turtle import window_width
import cv2
import numpy as np
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# 立体視のコード
# スキャンをせず、微分係数だけで演算するバージョン

def main_func(waitnum=1):

    image = 'photo' # これ以外（動画）は実質受け付けない
    video_file_name ='D0002060316_00000.mp4' # クルマ

    if image == 'normal_cam':
        filen = 'normal_cam'
        cam = cv2.VideoCapture(0) # PCのカメラ
        blurred_rate = 1.0
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        cam = cv2.VideoCapture(1) # USBカメラ
        blurred_rate = 1.0
    elif image == 'file':
        filen = 'Video_file'
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
        blurred_rate = 2.0
    elif image == 'photo':
        blurred_rate = 6
        left_photo  = "20220903_130929200_iOS.jpg" # テーブルなどの室内
        #right_photo = "20220903_130936446_iOS.jpg" # テーブルなどの室内 1cm
        right_photo = "20220903_130943000_iOS.jpg" # テーブルなどの室内 2cm
        #right_photo = "20220903_130948769_iOS.jpg" # テーブルなどの室内 3cm
        #right_photo = "20220903_130953706_iOS.jpg" # テーブルなどの室内 4cm
        #right_photo = "20220903_130959798_iOS.jpg" # テーブルなどの室内 5cm
        #right_photo = "20220903_131004597_iOS.jpg" # テーブルなどの室内 6cm


        left_photo  = "20220903_131052523_iOS.jpg" # 扇風機
        #right_photo = "20220903_131057994_iOS.jpg" # 扇風機 1cm
        right_photo = "20220903_131102820_iOS.jpg" # 扇風機 2cm
        #right_photo = "20220903_131108522_iOS.jpg" # 扇風機 3cm
        #right_photo = "20220903_131114917_iOS.jpg" # 扇風機 4cm
        #right_photo = "20220903_131119419_iOS.jpg" # 扇風機 5cm
        #right_photo = "20220903_131124568_iOS.jpg" # 扇風機 6cm
        #right_photo = "20220903_131129901_iOS.jpg" # 扇風機 7cm

        left_img  = np.float64(cv2.imread(left_photo))/256.
        #left_img = cv2.rotate(left_img, cv2.ROTATE_90_CLOCKWISE)
        right_img = np.float64(cv2.imread(right_photo))/256.
        #right_img = cv2.rotate(right_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        print('Please specify image')

    if image != 'photo':
        ret, img = cam.read() # ret, img = にしないと動かない
    y_size, x_size, c = left_img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)
    print(x_size_s, y_size_s)

    left_imgs  = cv2.resize(left_img,  (x_size_s, y_size_s))
    right_imgs = cv2.resize(right_img, (x_size_s, y_size_s))
    print(left_imgs.shape) # (y_size, x_size_s, 3)

    # 初期の位置合わせ用パラメータ
    max_shift_y = 30 # 偶数のこと 想定する最大ズレ量（ピクセル）
    max_shift_x = 30 # 偶数のこと 想定する最大ズレ量（ピクセル）
    x_size_w = x_size_s-max_shift_x
    y_size_w = y_size_s-max_shift_y

    # 初期の位置合わせ用ぼかしフィルタ
    gaussianSize = 31   # 奇数のこと、コンボリューションのカーネルサイズ
    gaussianWise_i = 8. # ぼかし広さ

    # 測距スキャンニング用ぼかしフィルタのカーネル。縦にぼかす
    kernelt = np.array([[0., 0.1,0.2,0.1, 0. ],\
                        [0.1,0.2,0.5,0.2, 0.1],\
                        [0.2,0.5,0.7,0.5, 0.2],\
                        [0.2,0.7,1., 0.7, 0.2],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.3,1., 1.5, 1., 0.3],\
                        [0.2,0.7,1., 0.7, 0.2],\
                        [0.2,0.5,0.7,0.5, 0.2],\
                        [0.1,0.2,0.5,0.2, 0.1],\
                        [0., 0.1,0.2,0.1, 0.]])
    kernelb = kernelt/np.sum(kernelt)
    gaussianWise_s = 2. # ぼかし広さ

    kernel = np.array([[-1., -1., 1., 1. ]])

    while True:
        if image != 'photo':
            ret, img = cam.read() # ret, img = にしないと動かない
        M = np.float32([[1,0,max_shift_y/2],[0,1,max_shift_x/2]])
        left_img_shift = cv2.warpAffine(left_imgs, M, (x_size_s, y_size_s))
        #left_img_win = left_img_shift[50:700, 50:700]
        left_img_win = left_img_shift[int(max_shift_y/2): int(y_size_s-max_shift_y/2), int(max_shift_x/2): int(x_size_s-max_shift_x/2)]
        print("Window size theory")
        print(x_size_w)
        print(y_size_w)
        y_size_p, x_size_p, c = left_img_win.shape
        print("Window size actual")
        print(x_size_p)
        print(y_size_p)
        left_img_win_blur_i = cv2.GaussianBlur(left_img_win, (gaussianSize,gaussianSize), gaussianWise_i)
        left_img_array_blur_i = np.float64(left_img_win_blur_i.reshape([x_size_w*y_size_w*3]))

        # 測距用スムージングフィルタ。本当はｙ方向にだけぼかしたい
        left_img_win_blur_s = cv2.GaussianBlur(left_img_win, (gaussianSize,gaussianSize), gaussianWise_s)
        left_img_win_blur_s = cv2.filter2D(left_img_win_blur_s, -1, kernelb)
        cv2.imshow('left', left_img_win_blur_s)
        key = cv2.waitKey(1)
        # 初期位置の合わせ込み
        print("finding initial posision")
        diff_min = 0.
        for x_shift in range(max_shift_x):
            for y_shift in range(max_shift_y):
                M = np.float32([[1,0,y_shift],[0,1,x_shift]])
                right_img_shift = cv2.warpAffine(right_imgs, M, (x_size_s, y_size_s))
                right_img_win = right_img_shift[int(max_shift_y/2): int(y_size_s-max_shift_y/2), int(max_shift_x/2): int(x_size_s-max_shift_x/2)]
                right_img_win_blur_i = cv2.GaussianBlur(right_img_win, (gaussianSize,gaussianSize), gaussianWise_i)
                right_img_array_blur_i = right_img_win_blur_i.reshape([x_size_w*y_size_w*3])
                diff = np.sum(abs(left_img_array_blur_i - right_img_array_blur_i))
                if diff_min == 0. or diff < diff_min:
                    diff_min = diff
                    x_shift_best = x_shift
                    y_shift_best = y_shift
        #x_shift_best = 1 # 15
        #y_shift_best = 0 # 21
        M = np.float32([[1,0,y_shift_best],[0,1,x_shift_best]])
        right_img_shift = cv2.warpAffine(right_imgs, M, (x_size_s, y_size_s))
        right_img_win = right_img_shift[int(max_shift_y/2): int(y_size_s-max_shift_y/2), int(max_shift_x/2): int(x_size_s-max_shift_x/2)]
        right_img_win_blur = cv2.GaussianBlur(right_img_win, (gaussianSize,gaussianSize), gaussianWise_s)
        right_img_win_blur = cv2.filter2D(right_img_win_blur, -1, kernelb)
        print(x_shift_best)
        print(y_shift_best)
        cv2.imshow('right', right_img_win_blur)
        key = cv2.waitKey(1)

        # x微分
        diff_img_left  = cv2.filter2D(left_img_win_blur_s, -1, kernel) + 0.5/256.
        diff_img_right = cv2.filter2D(right_img_win_blur,  -1, kernel) + 0.5/256.
        diff = diff_img_left + diff_img_right
        # 差分
        differ_img = right_img_win_blur - left_img_win_blur_s + 0.5/256.
        #img_dist = -0.07*diff/differ_img + 0.5
        img_dist = -0.8*differ_img/diff + 0.5
        img_dist = np.minimum(1.0, np.maximum(0., img_dist))

        print("Scanning distance")
        cv2.imshow('distance', img_dist)

        if waitnum == 0:
            print("Waiting key input...")
        key = cv2.waitKey(waitnum) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec) 引数waitnumをゼロにすると待ち続ける
        if key == 27 or image == 'photo': # when ESC key is pressed break
            break

    if image == 'photo':
        imgw = np.array(img_dist*256., np.int32)
        cv2.imwrite("dist_imgb.jpg", imgw,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        #print(img_dist)
    else:
        #cv2.waitKey(0)
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    main_func(0)
    
