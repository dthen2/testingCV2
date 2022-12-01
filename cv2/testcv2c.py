import cv2
import numpy as np
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

# cv2 で使うデータの構造については下記を参照
# http://makotomurakami.com/blog/2020/03/28/4232/

# 輪郭を抽出するプログラムを作った
# 行列演算を使わずに for 文を回しているので、遅いったらありゃしない
# testcv2.py の動画＆スパーステンソル変換バージョン･･･ファイルだけ作って手がついてない
# で、代わりに1次元に変換して、行列演算を試みた。
# dot 演算の前後を間違えると正方行列になり、しかもエラーメッセージが不適切で悩まされた
# 「スパース行列との引き算はできない」と表示され、それでデンスに変換しようともがいていたが、
# 実際の間違いは上記の通り、dot の前後が逆だったというお粗末だった。

conv_y = np.array([[ 1, 1, 1],\
                   [ 0, 0, 0],\
                   [-1,-1,-1]])

conv_x = np.array([[1, 0, -1],\
                   [1, 0, -1],\
                   [1, 0, -1]])
                 
def make_conv_matrix(img_size_x, img_size_y, conv):
    y_num, x_num = conv.shape
    x_even = 1 -x_num%2
    if x_even:
        print('Size of conv matrix must be odd row and col.')
    x_start = -x_num//2
    x_stop = -x_start + 1 - x_even
    y_even = 1 - y_num%2
    if y_even:
        print('Size of conv matrix must be odd row and col.')
    y_start = -y_num//2
    y_stop = -y_start + 1 - y_even
    data = []
    row_diff = []
    col_diff = []
    max_row_col = img_size_x * img_size_y * 3 -1
    for y in range(img_size_y):
        for x in range(img_size_x):
            for x_ in range(x_start, x_stop):
                for y_ in range(y_start, y_stop):
                    for c in range(3):
                        i = -x_start + x_
                        j = -y_start + y_
                        d = conv[j,i]
                        if d != 0:
                            row_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                            col_diff_n = max(0,min((y+y_)*img_size_x*3 + (x+x_)*3 + c, max_row_col))
                            row_diff.append(row_diff_n)
                            col_diff.append(col_diff_n)
                            data.append(d)
    conv_matrix = csr_matrix((data, (row_diff, col_diff)))#.toarray() #toarrayをつけるとデンスになっちゃう
    return conv_matrix


if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、2解明光のrunで照度が下がる？？？
    image = 'normal_cam'
    #image = 'file'
    #video_file_name = 'mov_hts-samp009.mp4' # 飛行機
    #video_file_name ='D0002160807_00000.mp4' # 街
    video_file_name ='D0002060316_00000.mp4' # クルマ

    if image == 'normal_cam':
        filen = 'normal_cam'
        cam = cv2.VideoCapture(0) # PCのカメラ
        blurred_rate = 1.0
        sh = 0.7
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        cam = cv2.VideoCapture(1) # USBカメラ
        blurred_rate = 1.0
        sh = 1
    elif image == 'file':
        filen = 'Video_file'
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
        blurred_rate = 2.0
        sh = 1
    else:
        print('Please specify image')

    ret, img = cam.read() # ret, img = にしないと動かない
    y_size, x_size, c = img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)
    print(x_size_s, y_size_s)

    imgs = cv2.resize(img, (x_size_s, y_size_s))
    print(imgs.shape) # (y_size, x_size_s, 3)

    reduction = 16
    amplifyer = 250

    # 変換行列作成。画像は1次元配列に変換した画像データを変換する。
    # https://note.nkmk.me/python-scipy-sparse-matrix-csr-csc-coo-lil/
    data = []
    row_xdiff = []
    row_ydiff = []
    col_xdiff = []
    col_ydiff = []
    max_row_col = x_size_s * y_size_s * 3 -1
    for y in range(y_size_s):
        for x in range(x_size_s):
            for c in range(3):
                rowcol_n = y*x_size_s*3 + x*3 + c
                row_ydiff_n = min((y+1)*x_size_s*3 + x*3 + c, max_row_col)
                col_xdiff_n = min(y*x_size_s*3 + (x+1)*3 + c, max_row_col)
                row_xdiff.append(rowcol_n)
                row_ydiff.append(row_ydiff_n)
                col_xdiff.append(col_xdiff_n)
                col_ydiff.append(rowcol_n)
                data.append(1)
    y_sift_matrix = csr_matrix((data, (row_ydiff, col_ydiff)))#.toarray() #toarrayをつけるとデンスになっちゃう
    x_sift_matrix = csr_matrix((data, (row_xdiff, col_xdiff)))#.toarray()

    while True:
        ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_array = imgs.reshape([x_size_s*y_size_s*3]) # 1次元配列に変換
        #img_array = img_array.astype(np.float32) # これ要らない
        img_xsift = np.array(x_sift_matrix.dot(img_array)) # 行列による変換の核心部分
        img_xdiff = abs(img_array - img_xsift)
        img_ysift = y_sift_matrix.dot(img_array)
        img_ydiff = abs(img_array - img_ysift)
        img_diff_array = (img_ydiff + img_xdiff)/reduction
        img_diff = img_diff_array.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        img_diff2 = np.maximum(0, img_diff - sh)*amplifyer
        # 画像データを実数にした後は、２５０分の１にして返さないといけない？？？
        img_draw = np.maximum(0, 0.1 + 0.5*np.maximum(0,imgs)/256.0 - np.maximum(0, img_diff2))
        cv2.imshow(filen, imgs)
        cv2.imshow('diff', img_diff)
        cv2.imshow('diff2', img_diff2)
        cv2.imshow('draw', img_draw)

        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            break

    #cv2.waitKey(0)
    cam.release()
    cv2.destroyAllWindows()
