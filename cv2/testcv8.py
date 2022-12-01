import cv2
import numpy as np
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# ピンボケを直すプログラム
# testcv7b.py をベースに、ベッセル関数の逆数から作ったコンボリューション関数使用
# 巨大な変換行列を作らないので、動画は不可だがコンボリューション行列の大きさに制限が無い

# default settings
convolution_window = 1.24
gain_v = 1. #3->1, 2.5->0.3, 3.5->0.2
center_val = 0.0
nana_photo = 'IMG_5527.jpg'
blur_picel = 3. # この値を4以上にするとメモリが不足する。ピクセル数がディスクリートのため、この値を少し変えるとおかしくなる。
blurred_rate_p = 2.0

map_dx = 0.1
# max_x = 5.
map_list = [-38.144, -36.310, -31.064, -23.120, -13.546, -3.588, \
              5.525, 12.758, 17.421, 19.247, 18.395, 15.391, 11.003, 6.093, 1.464, \
             -2.265, -4.739, -5.892, -5.909, -5.151, -4.053, -3.020, -2.342, -2.144, -2.379, -2.862, -3.329, -3.512, -3.209, -2.336, -0.945, \
              0.782, 2.584, 4.168, 5.278, 5.745, 5.521, 4.679, 3.397, 1.914, 0.477, \
             -0.705, -1.504, -1.883, -1.899, -1.674, -1.359, -1.097, -0.984, -1.049, -1.256, -1.513, -1.701, -1.709, -1.461, -0.941, -0.194, \
              0.677, 1.535, 2.241, 2.683, 2.796, 2.578, 2.088, 1.430, 0.727, 0.099, \
             -0.366, -0.627, -0.692, -0.612, -0.466, -0.335, -0.284, -0.345, -0.507, -0.728, -0.938, -1.064, -1.047, -0.858, -0.508, -0.043, \
              0.463, 0.926, 1.267, 1.436, 1.415, 1.224, 0.917, 0.564, 0.238, \
             -0.004, -0.128, -0.135, -0.055, \
              0.063, 0.162, 0.197, 0.140]
# max_x = 8
map_list = [ 49.812, 40.669, 17.075, \
            -11.130, -32.375, -38.282, -27.058, -3.930, \
             21.437, 39.227, 43.483, 34.126, 16.296, \
             -2.425, -15.424, -19.559, -15.764, -7.810, -0.093, \
              4.371, 4.846, 2.501, \
             -0.763, -3.500, -5.272, -6.416, -7.349, -7.980, -7.626, -5.498, -1.389, \
              3.895, 8.703, 11.253, 10.526, 6.829, 1.715, \
             -2.726, -4.837, -4.115, -1.357, \
              1.840, 3.898, 3.983, 2.273, \
             -0.299, -2.632, -4.010, -4.347, -4.028, -3.533, -3.088, -2.562, -1.663, -0.246, \
              1.459, 2.893, 3.452, 2.852, 1.347, \
             -0.355, -1.432, -1.367, -0.219, \
              1.417, 2.718, 3.054, 2.283, 0.780, \
             -0.804, -1.888, -2.235, -1.996, -1.546, -1.218, -1.112, -1.084, -0.903, -0.449, \
              0.161, 0.632, 0.678, 0.227, \
             -0.499, -1.071, -1.090, -0.415, \
              0.733, 1.870, 2.491, 2.319, 1.444, 0.259, \
             -0.749, -1.241, -1.164, -0.732, -0.268, -0.010, -0.006]

def anti_func(x):
    max_n = len(map_list)-1
    n = int(abs(x)/map_dx)
    if n > max_n:
        val = 0.
    else:
        val = map_list[n]
    return val

def make_conv_org3(blur_dist):
    radi = 2*convolution_window
    size = int(blur_dist*radi)
    even = 1 -size%2
    if even:
        size = size-1
    cent = int(blur_dist*radi/2.)
    conv = np.zeros( (size, size) )
    gain = gain_v/256.
    dist_fact = 0.3 # 4 for max_x=5
    for x in range(size):
        for y in range(size):
            dist = np.sqrt((x-cent)**2 + (y-cent)**2)
            func = gain*anti_func(dist*dist_fact)
            conv[x,y] = func
    conv[cent, cent] += center_val
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
    for y in range(img_size_y):
        print('\r  progress %.1f%%' % (100.*(y+1.)/img_size_y), end = "")
        img_shape = new_image.reshape([img_size_y, img_size_x, 3]) # 元の配列の形に戻す。
        img_shape = np.maximum(0., img_shape)
        cv2.imshow('shape', img_shape)
        key = cv2.waitKey(1)
        for x in range(img_size_x):
            for x_ in range(x_start, x_stop):
                for y_ in range(y_start, y_stop):
                    for c in range(3):
                        i = -x_start + x_
                        j = -y_start + y_
                        rowcol_n = y*img_size_x*3 + x*3 + c
                        col_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                        new_image[rowcol_n] += img[col_diff_n]*conv[j,i]
    print("")
    return new_image

def main_func(new_file_name, waitnum=1):

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
    conv = make_conv_org3(blur_dist)
    #print('Constructing conversion matrix')
    #anti_blur_matrix  = make_conv_matrix(x_size_s, y_size_s, conv)

    while True:
        if image != 'photo':
            ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_array = imgs.reshape([x_size_s*y_size_s*3]) # 1次元配列に変換
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

        if waitnum == 0:
            print("Waiting key input...")
        key = cv2.waitKey(waitnum) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec) 引数waitnumをゼロにすると待ち続ける
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
    gain_v = 0.008 #
    center_val = 0.0
    nana_photo = 'IMG_5527b.jpg'
    blur_picel = 12.    # 

#    blurred_rate_p = 4.0
#    main_func("sharp_v8.jpg", 0)
#    blurred_rate_p = 4.0
#    main_func("sharp_v8b.jpg")
    blurred_rate_p = 3.6
    main_func("sharp_v8c.jpg")
    blurred_rate_p = 3.3
    main_func("sharp_v8d.jpg")
    blurred_rate_p = 3.0
    main_func("sharp_v8e.jpg")
    blurred_rate_p = 2.6
    main_func("sharp_v8f.jpg")
    blurred_rate_p = 2.3
    main_func("sharp_v8g.jpg")
    blurred_rate_p = 2.0
    main_func("sharp_v8h.jpg")
    
