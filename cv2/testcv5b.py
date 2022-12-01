import cv2
import numpy as np
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# ピンボケを直すプログラム
# 輪郭を抽出するtestcv2c.py をベースに作成
# 行列演算を使わずに for 文を回しているので、遅いったらありゃしない

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


def make_conv_matrix(img_size_x, img_size_y, conv):
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
    data = []
    row_diff = []
    col_diff = []
    max_row_col = img_size_x * img_size_y * 3 -1
    for y in range(img_size_y):
        prog = 100.*(y+1)/img_size_y
        perc = abs(prog/10 - int(prog/10))
        if perc < 0.02:
            print('progress %.0f' % prog)
#        perc = (y*100) % 10
#        if perc != 0:
#            print('progress %.0f' % (100.*y/img_size_y))
        for x in range(img_size_x):
            for x_ in range(x_start, x_stop):
                for y_ in range(y_start, y_stop):
                    for c in range(3):
                        i = -x_start + x_
                        j = -y_start + y_
                        d = conv[j,i]
                        if d != 0:
                            rowcol_n = y*img_size_x*3 + x*3 + c
                            #row_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                            col_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                            row_diff.append(rowcol_n)
                            col_diff.append(col_diff_n)
                            data.append(d)
                            #print('i,j = %d, %d', i, j )
    print('Constructing spars conversion matrix') 
    conv_matrix = csr_matrix((data, (row_diff, col_diff)))#.toarray() #toarrayをつけるとデンスになっちゃう
    return conv_matrix

def main_func():
#if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、2解明光のrunで照度が下がる？？？
    image = 'normal_cam'
    #image = 'file'
    image = 'photo'
    #video_file_name = 'mov_hts-samp009.mp4' # 飛行機
    #video_file_name ='D0002160807_00000.mp4' # 街
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
    print('Constructing conversion matrix')
    anti_blur_matrix  = make_conv_matrix(x_size_s, y_size_s, conv)

    while True:
        if image != 'photo':
            ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_array = imgs.reshape([x_size_s*y_size_s*3]) # 1次元配列に変換
        img_shape = anti_blur_matrix.dot(img_array)
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

        key = cv2.waitKey(0) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27 or image == 'photo': # when ESC key is pressed break
            break

    if image == 'photo':
        imgw = np.array(img_shape*256., np.int32)
        cv2.imwrite('hoges.jpg', imgw,  [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(img_shape)
    else:
        #cv2.waitKey(0)
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    convolution_window = 1.24
    dist_limit = 2.
    gain_v = 0.1 #3->1, 2.5->0.3, 3.5->0.2
    center_val = 0.004
    nana_photo = 'IMG_5527b.jpg'
    blurred_rate_p = 8.0
    blur_picel = 4. # この値を4以上にするとメモリが不足する。ピクセル数がディスクリートのため、この値を少し変えるとおかしくなる。
    main_func()
