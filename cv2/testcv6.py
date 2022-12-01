import cv2
import numpy as np
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# ピンボケを直すプログラム
# testcv5b.py をベースに作成
# 巨大な変換行列を作らないので、動画は不可だがコンボリューション行列の大きさに制限が無い

# default settings
convolution_window = 1.24
dist_limit = 3.
gain_v = 1. #3->1, 2.5->0.3, 3.5->0.2
center_val = 0.0
nana_photo = 'IMG_5527.jpg'
blur_picel = 3. # この値を4以上にするとメモリが不足する。ピクセル数がディスクリートのため、この値を少し変えるとおかしくなる。
blurred_rate_p = 2.0

def make_conv_org(blur_dist):
    radi = 2*convolution_window
    size = int(blur_dist*radi)
    even = 1 -size%2
    if even:
        size = size-1
    cent = int(blur_dist*radi/2.)
    conv = np.zeros( (size, size) )
    gain = gain_v/256.  
    for x in range(size):
        for y in range(size):
            dist_inv  = blur_dist - np.sqrt((x-cent)**2 + (y-cent)**2)
            if dist_inv > 0.:
                if dist_inv > dist_limit:
                    func = 0.
                elif dist_inv < 1.:
                    dist_inv = 1.
                    func = gain/(dist_inv)
                else:
                    func = gain/(dist_inv)
                conv[x,y] = func
            elif dist_inv < 0.:
                if dist_inv < -dist_limit:
                    func =0.
                elif dist_inv > -1.:
                    dist_inv = -1.
                    func = gain/(dist_inv)
                else:
                    func = gain/(dist_inv)
                conv[x,y] = func
            elif dist_inv == 0.:
                func = 0.
            else: # ここは来ないはず
                func = gain/(dist_inv)
                conv[x,y] = func
    conv[cent, cent] = center_val
    print(conv)
    return conv


def conversion(img, img_size_x, img_size_y, conv):
    print('Conversing')
    y_num, x_num = conv.shape
    x_even = 1 -x_num%2
    if x_even:
        print('Size of conv matrix must be odd row and col.')
    x_start = -(x_num//2)
    x_stop = -x_start + 1 - x_even
    y_even = 1 - y_num%2
    if y_even:
        print('Size of conv matrix must be odd row and col.')
    y_start = -(y_num//2)
    y_stop = -y_start + 1 - y_even
    max_row_col = img_size_x * img_size_y * 3 -1
    new_image = np.zeros(max_row_col+1)
    counter = 0
    for y in range(img_size_y):
        counter += 1
        if counter > ((img_size_y)/10.-0.49) :
            counter = 0
            print('  progress %.0f%%' % (100*(y+1)/img_size_y))
        for x in range(img_size_x):
            for x_ in range(x_start, x_stop):
                for y_ in range(y_start, y_stop):
                    for c in range(3):
                        i = -x_start + x_
                        j = -y_start + y_
                        rowcol_n = y*img_size_x*3 + x*3 + c
                        col_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                        new_image[rowcol_n] += img[col_diff_n]*conv[j,i]
    return new_image

def main_func(new_file_name):

    image = 'photo'
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
        blurred_rate = blurred_rate_p
        filen = nana_photo # ナナとワーム。床のマットの輪郭がくっきり
        img = cv2.imread(filen) 
        blur_dist = blur_picel  # この値を4以上にするとメモリが不足する。ピクセル数がディスクリートのため、この値を少し変えるとおかしくなる。
    else:
        print('Please specify image')

    if image != 'photo':
        ret, img = cam.read() # ret, img = にしないと動かない
    y_size, x_size, c = img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)
    print(x_size_s, y_size_s)

    imgs = cv2.resize(img, (x_size_s, y_size_s))
    print(imgs.shape) # (y_size, x_size_s, 3)

    print('Constructing anti blur matrix')
    conv = make_conv_org(blur_dist)
    #print('Constructing conversion matrix')
    #anti_blur_matrix  = make_conv_matrix(x_size_s, y_size_s, conv)

    while True:
        if image != 'photo':
            ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_array = imgs.reshape([x_size_s*y_size_s*3]) # 1次元配列に変換
        print(new_file_name)
        img_shape = conversion(img_array, x_size_s, y_size_s, conv)
        img_shape = img_shape.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        img_shape = np.maximum(0., img_shape)
        img_shape2 = img_shape*10.
        img_shape3 = img_shape/10.0
        #img_shape4 = img_shape*1000.0
        cv2.imshow('shape', img_shape)
        cv2.imshow('shape2', img_shape2)
        cv2.imshow('shape3', img_shape3)
        #cv2.imshow('shape4', img_shape4)
        cv2.imshow(filen, imgs)

        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec) 引数をゼロにすると待ち続ける
        if key == 27 or image == 'photo': # when ESC key is pressed break
            break

    if image == 'photo':
        imgw = np.array(img_shape*256., np.int32)
        cv2.imwrite(new_file_name, imgw,  [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(img_shape)
    else:
        #cv2.waitKey(0)
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    convolution_window = 1.24
    dist_limit = 2.
    gain_v = 0.04 #3->1, 2.5->0.3, 3.5->0.2
    center_val = 0.008
    nana_photo = 'IMG_5527b.jpg'
    blur_picel = 8. #

    blurred_rate_p = 2.8
    main_func("sharp_v6k.jpg")
    blurred_rate_p = 2.7
    main_func("sharp_v6l.jpg")
    blurred_rate_p = 2.6
    main_func("sharp_v6m.jpg")
    blurred_rate_p = 2.5
    main_func("sharp_v6n.jpg")
    blurred_rate_p = 2.4
    main_func("sharp_v6o.jpg")
    blurred_rate_p = 2.3
    main_func("sharp_v6p.jpg")
    blurred_rate_p = 2.2
    main_func("sharp_v6q.jpg")
    blurred_rate_p = 2.1
    main_func("sharp_v6r.jpg")
    blurred_rate_p = 2.
    main_func("sharp_v6s.jpg")
