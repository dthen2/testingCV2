import copy
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
# さらに行列を自動生成するアルゴリズム制作したのが、testcv2d.py
# 時間微分も入れて、全部乗せ
                 
# 変換行列作成。画像は1次元配列に変換した画像データを変換する。
# https://note.nkmk.me/python-scipy-sparse-matrix-csr-csc-coo-lil/
# conv は奇数行奇数列の numpy array であること
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
    conv_matrix = csr_matrix((data, (row_diff, col_diff)))#.toarray() #toarrayをつけるとデンスになっちゃう
    #conv_matrix = csr_matrix((data, (col_diff, row_diff)))
    return conv_matrix

# 色変換行列作成
# conv は３行３列の numpy array である必要
def make_color_matrix(img_size_x, img_size_y, conv):
    data = []
    row_diff = []
    col_diff = []
    max_row_col = img_size_x * img_size_y * 3 -1
    for y in range(img_size_y):
        for x in range(img_size_x):
            for c_out in range(3):
                for c_in in range(3):
                    d = conv[c_in,c_out]
                    if d != 0:
                        rowcol_n =   max(0,min(y*img_size_x*3 + x*3 + c_out, max_row_col))
                        col_diff_n = max(0,min(y*img_size_x*3 + x*3 + c_in,  max_row_col))
                        row_diff.append(rowcol_n)
                        col_diff.append(col_diff_n)
                        data.append(d)
    conv_matrix = csr_matrix((data, (row_diff, col_diff)))#.toarray() #toarrayをつけるとデンスになっちゃう
    #conv_matrix = csr_matrix((data, (col_diff, row_diff)))
    return conv_matrix

if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、2回目のrunで照度が下がる？？？
    image = 'normal_cam'
    #image = 'file'
    video_file_name = 'planes.mp4' # 飛行機
    video_file_name ='town.mp4' # 街
    video_file_name ='cars.mp4' # クルマ
    video_file_name ='IMG_2712.mov' # ナナ

    if image == 'normal_cam':
        filen = 'normal_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(0) # PCのカメラ
        blurred_rate = 1.0
        sh = 0.5
        # 時間微分用
        sht = 30
        amp = 5
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(1) # USBカメラ
        blurred_rate = 1.0
        sh = 0.5
        # 時間微分用
        sht = 30
        amp = 5
    elif image == 'file':
        filen = 'Video_file'
        frame_rate = 24.0
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
        blurred_rate = 3.0
        sh = 0.8#0.1
        # 時間微分用
        sht = 30
        amp = 5
    else:
        print('Please specify image')

    ret, img = cam.read() # ret, img = にしないと動かない
    y_size, x_size, c = img.shape
    print(x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)

    imgs = cv2.resize(img, (x_size_s, y_size_s))

    # 動画保存
    record_size = (x_size_s, y_size_s)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('record_video.mp4', fmt, frame_rate, record_size)

    reduction = 0.5
    amplifyer = 50.0

    conv_y = np.array([[.5,.5,.5,.5, .5],\
                       [ 1, 1, 1, 1, 1],\
                       [ 0, 0, 0, 0, 0],\
                       [-1,-1,-1,-1,-1],\
                       [-.5,-.5,-.5,-.5, -.5]])
    conv_x = np.array([[.5, 1, 0, -1,-.5],\
                       [.5, 1, 0, -1,-.5],\
                       [.5, 1, 0, -1,-.5],\
                       [.5, 1, 0, -1,-.5],\
                       [.5, 1, 0, -1,-.5]])

    conv_y = np.array([[ 1, 1, 1, 1, 1],\
                       [ 0, 0, 0, 0, 0],\
                       [-1,-1,-1,-1,-1]]) # こっちの方がよさそう
    conv_x = np.array([[1, 0, -1],\
                       [1, 0, -1],\
                       [1, 0, -1],\
                       [1, 0, -1],\
                       [1, 0, -1]]) # こっちの方がよさそう

    #conv_y = np.array([[ 1, 1, 1, 1, 1, 1, 1],\
    #                   [ 0, 0, 0, 0, 0, 0, 0],\
    #                   [-1,-1,-1,-1,-1,-1,-1]]) # 次元増やしてみた。繋がった線としての検出感度上がってＳＮも良くなるが、細かい絵が潰れる
    #conv_x = np.array([[1, 0, -1],\
    #                   [1, 0, -1],\
    #                   [1, 0, -1],\
    #                   [1, 0, -1],\
    #                   [1, 0, -1],\
    #                   [1, 0, -1],\
    #                   [1, 0, -1]]) # 次元増やしてみた。繋がった線としての検出感度上がってＳＮも良くなるが、細かい絵が潰れる

    conv_xy = np.array([[ 0, 1,.5,.2, 0],\
                        [-1, 0, 1,.5,.2],\
                        [-.5,-1, 0, 1,.5],\
                        [ -.2,-.5,-1, 0, 1],\
                        [ 0, -.2, -.5,-1, 0]])
    conv_yx = np.array([[ 0,.2,.5, 1, 0],\
                        [.2,.5, 1, 0,-1],\
                        [.5, 1, 0,-1,-.5],\
                        [ 1, 0,-1,-.5,-.2],\
                        [ 0,-1,-.5,-.2, 0]])

    conv_xy = np.array([[ 0, 1, 0, 0, 0],\
                        [-1, 0, 1, 0, 0],\
                        [ 0,-1, 0, 1, 0],\
                        [ 0, 0,-1, 0, 1],\
                        [ 0, 0, 0,-1, 0]])*1.5 # こっちの方がよさそう
    conv_yx = np.array([[ 0, 0, 0, 1, 0],\
                        [ 0, 0, 1, 0,-1],\
                        [ 0, 1, 0,-1, 0],\
                        [ 1, 0,-1, 0, 0],\
                        [ 0,-1, 0, 0, 0]])*1.5 # こっちの方がよさそう

    #conv_xy = np.array([[ 0, 1, 0, 0, 0, 0, 0],\
    #                    [-1, 0, 1, 0, 0, 0, 0],\
    #                    [ 0,-1, 0, 1, 0, 0, 0],\
    #                    [ 0, 0,-1, 0, 1, 0, 0],\
    #                    [ 0, 0, 0,-1, 0, 1, 0],\
    #                    [ 0, 0, 0, 0,-1, 0, 1],\
    #                    [ 0, 0, 0, 0, 0,-1, 0]])*1.5 # 次元増やしてみた。繋がった線としての検出感度上がってＳＮも良くなるが、細かい絵が潰れる

    #conv_yx = np.array([[ 0, 0, 0, 0, 0, 1, 0],\
    #                    [ 0, 0, 0, 0, 1, 0,-1],\
    #                    [ 0, 0, 0, 1, 0,-1, 0],\
    #                    [ 0, 0, 1, 0,-1, 0, 0],\
    #                    [ 0, 1, 0,-1, 0, 0, 0],\
    #                    [ 1, 0,-1, 0, 0, 0, 0],\
    #                    [ 0,-1, 0, 0, 0, 0, 0]])*1.5 # 次元増やしてみた。繋がった線としての検出感度上がってＳＮも良くなるが、細かい絵が潰れる

    # 線検出の場合は下記を用いる。変換の性質上、ノイズに敏感で使えそうにない
    conv_ly = np.array([[-1,-1,-1,-1,-1],\
                        [ 2, 2, 2, 2, 2],\
                        [-1,-1,-1,-1,-1]])*3
    conv_lx = np.array([[-1, 2, -1],\
                        [-1, 2, -1],\
                        [-1, 2, -1],\
                        [-1, 2, -1],\
                        [-1, 2, -1]])*3
    conv_lxy = np.array([[ 1,-1, 0, 0, 0],\
                         [-1, 2,-1, 0, 0],\
                         [ 0,-1, 2,-1, 0],\
                         [ 0, 0,-1, 2,-1],\
                         [ 0, 0, 0,-1, 1]])*3
    conv_lyx = np.array([[ 0, 0, 0,-1, 1],\
                         [ 0, 0,-1, 2,-1],\
                         [ 0,-1, 2,-1, 0],\
                         [-1, 2,-1, 0, 0],\
                         [ 1,-1, 0, 0, 0]])*3

    print('Constructing conversion matrix y-edge')
    y_sift_matrix  = make_conv_matrix(x_size_s, y_size_s, conv_y)
    print('Constructing conversion matrix x-edge')
    x_sift_matrix  = make_conv_matrix(x_size_s, y_size_s, conv_x)
    print('Constructing conversion matrix yx-edge')
    yx_sift_matrix = make_conv_matrix(x_size_s, y_size_s, conv_yx)
    print('Constructing conversion matrix xy-edge')
    xy_sift_matrix = make_conv_matrix(x_size_s, y_size_s, conv_xy)

    conv_blurr = np.array( [[ 0, 1, 1, 1, 0],\
                            [ 1, 1, 1, 1, 1],\
                            [ 1, 1, 1, 1, 1],\
                            [ 1, 1, 1, 1, 1],\
                            [ 0, 1, 1, 1, 0]])*0.04 # ピーク抽出用ぼかしフィルター。左記の0.04はかなりいい線いってる数値 

    #conv_blurr = np.array( [[ 0, 1, 1, 1, 1, 1, 0],\
    #                        [ 1, 1, 1, 1, 1, 1, 1],\
    #                        [ 1, 1, 1, 1, 1, 1, 1],\
    #                        [ 1, 1, 1, 1, 1, 1, 1],\
    #                        [ 1, 1, 1, 1, 1, 1, 1],\
    #                        [ 1, 1, 1, 1, 1, 1, 1],\
    #                        [ 0, 1, 1, 1, 1, 1, 0]])*0.025 # ピーク抽出用ぼかしフィルター。左記の0.025はかなりいい線いってる数値 

    print('Constructing blurring conversion matrix')
    blurr_matrix = make_conv_matrix(x_size_s, y_size_s, conv_blurr)
    # ぼかしたのを引くと、ピークが残る。という方法でエッジのピークを抽出
    # SciPy に、何故か単位行列生成のメソッドが無い・・・

    print('Constructing color conversion matrix to white and black')
    conv_WB = np.array([[1,1,1],\
                        [1,1,1],\
                        [1,1,1]])/3.0
    conv_WB_matrix = make_color_matrix(x_size_s, y_size_s, conv_WB)

    img_last = cv2.resize(img, (x_size_s, y_size_s))
    ceiling_value = 250

    while True:
        ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        img_array = imgs.reshape([x_size_s*y_size_s*3])/256.0 # 1次元配列に変換。ここで２５６分の１に
        img_xdiff = (np.array(x_sift_matrix.dot(img_array)))**2 # 行列による変換の核心部分
        img_ydiff = (np.array(y_sift_matrix.dot(img_array)))**2
        img_xydiff = (np.array(xy_sift_matrix.dot(img_array)))**2
        img_yxdiff = (np.array(yx_sift_matrix.dot(img_array)))**2
        img_diff_array = (img_ydiff + img_xdiff + img_xydiff + img_yxdiff)/reduction
        img_diff_array = np.array(conv_WB_matrix.dot(img_diff_array)) # 検出したエッジイメージを白黒にする
        img_diff_a_peak = np.maximum(0,img_diff_array - np.array(blurr_matrix.dot(img_diff_array)))*7.0 #ぼかしたのを引くと、ピークが残る
        img_diff = img_diff_array.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        img_diff_peak = img_diff_a_peak.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        img_diff2 = np.maximum(0, img_diff_peak - sh)*amplifyer
        # 漫画風変換。画像データを実数にした後は、２５０分の１にして返さないといけない？？？
        img_draw = np.maximum(0, 0.1 + 0.5*np.maximum(0,imgs)/256.0 - np.maximum(0, img_diff2))
        cv2.imshow(filen, imgs)
        cv2.imshow('diff', img_diff)
        cv2.imshow('diff2', img_diff2)
        cv2.imshow('draw', img_draw)
        cv2.imshow('peak', img_diff_peak)

        # 時間微分用
        # 整数のままでハンドリングできる場合は、２５６分の１は要らない
        img_diff_v = abs(np.array(imgs,dtype='int16') - np.array(img_last,dtype='int16')) - sht
        img_diff_c = np.minimum(np.maximum(img_diff_v*amp, 0), ceiling_value)
        img_diff_t = np.array(img_diff_c,dtype='uint8')
        cv2.imshow('diff_time', img_diff_t)
        img_last = copy.copy(imgs)

        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            break
        
        img_draw = np.array(np.minimum(255, np.maximum(0, 256*img_draw)),dtype='uint8')
        writer.write(img_draw) # 画像を1フレーム分として書き込み

    writer.release() # ファイルを閉じる

    #cv2.waitKey(0)
    cam.release()
    cv2.destroyAllWindows()
