import copy
import math
import cv2
import numpy as np
import random
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from matplotlib import pylab as plt
import point_cloud5 as pcl
import id_network as ids
import fourier_trans as frt

colors = [
  (0, 0, 255),
#  (0, 64, 255),
  (0, 128, 255),
#  (0, 192, 255),
  (0, 255, 255),
#  (0, 255, 192),
  (0, 255, 128),
#  (0, 255, 64),
  (0, 255, 0),
#  (64, 255, 0),
  (128, 255, 0),
#  (192, 255, 0),
  (255, 255, 0),
#  (255, 192, 0),
  (255, 128, 0),
#  (255, 64, 0),
  (255, 0, 0),
#  (255, 0, 64),
  (255, 0, 128),
#  (255, 0, 192),
  (255, 0, 255),
#  (192, 0, 255),
  (128, 0, 255),
#  (64, 0, 255),
]

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

# pict2points.py では、輪郭を自作フィルターで抽出、点群変換し、自作アルゴリズムによって分割とソートを行った。
# pict2points2.py では、輪郭をCannyで抽出後、点群として抽出し、参照画像とのマッチング比較を行った。左記マッチングとは別途、opencvによる分割も試した
# pict2points3.py では、輪郭をCannyで抽出し、点群として抽出後、そのギザギザ等を評価する。
# pict2points4.py では、動画にも対応し、自作エッジ検出を廃止。ギザギザ評価に、画像の縮小率(blurred_rate)変更への対応も入れた。

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

# 点群への変換
# 抽出した輪郭（img_draw の非ゼロ）の座標と、強度（img_diff）をリストにして返す
# img_diff, img_diff はnumpy配列である事。
# cont_color はデータ構造を識別するため。Trueならカラー、Falseなら白黒、または２値
def scan_contor(x_size_s, y_size_s, img_cont, img_diff, cont_color=True):
    coordi_list = []
    data_list = []
    for y in range(y_size_s):
        for x in range(x_size_s):
            #row = y*x_size_s*3 + x*3
            if cont_color:
                point = img_cont[y, x, 1]
            else:
                point = img_cont[y,x]
            if point: 
                coordi_list.append([float(x), float(y_size_s - y)])
                data_list.append(float(img_diff[y, x, 0]))
    return coordi_list, data_list

def cloudlength(cloud):
    return(cloud.len)

# 点の数が多い順に点群のリストをソートする
# さらに、点の数が少ない点群を切り捨てる
# clouds は点群のリスト
def sort_group_clouds(clouds):
    clouds.sort(key=cloudlength, reverse=True)
    nums = []
    num_sum = 0
    num_group = 0
    for cloud in clouds:
        nums.append(cloud.len)
        num_sum += cloud.len
        num_group += 1
        num_mean = num_sum/num_group
    num_sh = num_mean + 0.150*np.std(nums)    # 少ないとする判定基準。点の数の平均値＋点の数分布の３σ
    i = 0
    for cloud in clouds:
        i += 1
        if cloud.len < num_sh:
            break
    return clouds[:i], clouds[i:]

# 検出した輪郭を、点群毎に色分けして表示する
def create_colored_contors(x_size_s, y_size_s, img_diff, new_clouds, discarded):
    new_img = copy.copy(img_diff)
    i = 0
    for new_cloud in new_clouds:
        for xsd in new_cloud.xs:
            y = max(0, min(y_size_s-1, y_size_s - int(xsd[1])))
            x = max(0, min(x_size_s-1, int(xsd[0])))
            new_img[y, x] = colors[i]
            #new_img[xsd[1], xsd[0], 1] = colors[i][1]
            #new_img[xsd[1], xsd[0], 2] = colors[i][2]
        i += 1
        if i == (len(colors)-1):
            i = 0
    for d_cloud in discarded:
        for xsd in d_cloud.xs:
            new_img[y, x] = (0,0,0)
    return new_img


def plotdata(ax,closest):
    xsd=closest.xs
    refp=closest.ref
    rot = closest.rotation_best[0]
    sin = math.sin(rot)
    cos = math.cos(rot)
    sift = closest.sift_best
    magnify = closest.magnify_best
    centre_ref = closest.size_ref["centre"]
    for i in range(len(xsd)):
        x = (xsd[i][0] + sift[0])*magnify[0]*cos - (xsd[i][1] + sift[1])*magnify[1]*sin
        y = (xsd[i][1] + sift[1])*magnify[1]*cos + (xsd[i][0] + sift[0])*magnify[0]*sin
        ax.scatter(x,y, c="black", s=7.0, marker='o')
    for i in range(len(refp)):
        xr = refp[i][0] - centre_ref[0]
        yr = refp[i][1] - centre_ref[1]
        ax.scatter(xr,yr, c="red", s=5.0, marker='o')

######################### ###########
            # 参照データとの比較
def point_matching(coordi_list_ref, coordi_list):
    #imgs_ref = cv2.resize(img_ref, (x_size_sr, y_size_sr))
    #img_canny_ref = cv2.Canny(imgs_ref, 100, 200)
    #coordi_list_ref, data_list_ref = scan_contor(x_size_sr, y_size_sr, img_canny_ref, imgs_ref,cont_color=False)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.show()
    closest = pcl.Closest(2, coordi_list_ref, coordi_list) #ref の方に広い視野の画像、xsの方に見つけたい部品を入れた方がうまくいく 
    closest.magnify      = [0.4, 0.4] # ちょっとズル
    closest.magnify_best = [0.4, 0.4]
    mat = 0.0
    fig.show()
    resolution = 2. # 2.0
    print('Point matching started.')
    # 乱数による探索
    for i in range(5):
        plt.cla()
        plotdata(ax,closest)
        fig.show()
        plt.pause(0.01) #これが無いと何故か表示されない
        mat = closest.closest_point_search( search_num=20, resolution=resolution, matching_func_best=mat)
        print('mat=',mat)

    # ここから、グラジェントに基づいて行う
    resolution = 1.5 # 1.5 分解能変更
    mat = closest.closest_point_evaluation(resolution) # 分解能を変えたらmatを再評価
    cloud = pcl.PointCloud(2, xs=coordi_list)
    cloud.points_line_sort()
    closest.trans_best_to_default()
    dist_mean = cloud.dist_mean
    print('dist_mean =', dist_mean)
    rot_scale = dist_mean/cloud.cloud_radius*0.0001
    x_scale   = dist_mean*0.0001
    mag_scale = dist_mean/cloud.cloud_radius*0.0001
    d_x = [0.0, 0.0]
    d_rot = [0.0, 0.0]
    d_mag = [0.0, 0.0]
    for i in range(10):
        plt.cla()
        plotdata(ax,closest)
        fig.show()
        plt.pause(0.01) #これが無いと何故か表示されない
        diff_rot, diff_x, diff_mag = closest.closest_point_diff(resolution, rot_scale, x_scale, mag_scale)
        diff_x_scale = 0.0
        diff_rot_scale = 0.0
        diff_mag_scale = 0.0
        print('diff', diff_rot, diff_x, diff_mag)
        for n in range(2):
            diff_x_scale += abs(diff_x[n])
            diff_rot_scale += abs(diff_rot[n])
            diff_mag_scale += abs(diff_mag[n])
        for n in range(2):
            d_x[n]   = x_scale*diff_x[n]/diff_x_scale*2000. #20.
            d_rot[n] = rot_scale*diff_rot[n]/diff_rot_scale*20000. # 400.
            d_mag[n] = mag_scale*diff_mag[n]/diff_mag_scale*20000. # 200.
        mat, is_converge = closest.closest_point_approach( resolution, mat, d_x, d_rot, d_mag)
        closest.trans_best_to_default()
        print('d_x=', d_x)
        print('d_rot=', d_rot)
        print('d_mag=', d_mag)
        print(mat, is_converge)
        if is_converge:
            break
    plt.cla()
    plotdata(ax,closest)
    fig.show()
    plt.pause(0.01) #これが無いと何故か表示されない
    print('Ended')
    plt.show()
    #################



##########################################################################
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
    #pict_file_name = 'raiderman.jpeg'
    #pict_file_name = 'IMG_3195.jpg' # 手に持ったスパナ
    #pict_file_name = 'IMG_3199_.jpg' # フカダがスパナを持つ画像

    #pict_ref_file_name = 'spanner2.jpg'

    if image == 'normal_cam':
        filen = 'normal_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(0) # PCのカメラ
        ret, img = cam.read() # ret, img = にしないと動かない
        blurred_rate = 5.0
        sh = 1.5
        # 時間微分用
        sht = 30
        amp = 5
    elif image == 'jetson_nano_web_cam':
        filen = 'jetson_nano_web_cam'
        frame_rate = 5.0
        cam = cv2.VideoCapture(1) # USBカメラ
        ret, img = cam.read() # ret, img = にしないと動かない
        blurred_rate = 1.0
        sh = 0.5
        # 時間微分用
        sht = 30
        amp = 5
    elif image == 'file':
        filen = 'Video_file'
        frame_rate = 24.0
        cam = cv2.VideoCapture(video_file_name) # .mp4 ビデオファイル
        ret, img = cam.read() # ret, img = にしないと動かない
        blurred_rate = 3.0
        sh = 0.8#0.1
        # 時間微分用
        sht = 30
        amp = 5
    elif image == 'picture': # 静止画
        # https://peaceandhilightandpython.hatenablog.com/entry/2015/12/23/214840
        filen = 'picture_file'
        #frame_rate = 24.0
        img = cv2.imread(pict_file_name) # .画像ファイル
        #img_ref = cv2.imread(pict_ref_file_name)
        #blurred_ref = 10.0
        blurred_rate = 2.0 # 2.0=raiderman #5.0=nana # 3.0
        sh = 3.0 # 2.0
        #y_size_r, x_size_r, cc = img_ref.shape
        #x_size_sr = int(x_size_r/blurred_ref)
        #y_size_sr = int(y_size_r/blurred_ref)
        # 参照データとの比較開始
        #imgs_ref = cv2.resize(img_ref, (x_size_sr, y_size_sr))
        #img_canny_ref = cv2.Canny(imgs_ref, 150, 250)
        #cv2.imshow(pict_ref_file_name, img_canny_ref)
        #key = cv2.waitKey(1)
        #coordi_list_ref, data_list_ref = scan_contor(x_size_sr, y_size_sr, img_canny_ref, imgs_ref,cont_color=False)
        # 時間微分用
        #sht = 30
        #amp = 5
    else:
        print('Please specify image')
    
    y_size, x_size, c = img.shape
    print('Picture size =', x_size, y_size)
    x_size_s = int(x_size/blurred_rate)
    y_size_s = int(y_size/blurred_rate)

    if image != 'picture':
        # 動画保存準備
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer  = cv2.VideoWriter('record_video.mp4',  fmt, frame_rate, (x_size_s, y_size_s))
        writer2 = cv2.VideoWriter('record_video2.mp4', fmt, frame_rate, (x_size_s, y_size_s))
    
    imgs = cv2.resize(img, (x_size_s, y_size_s))
    print('Reduced picture size =', x_size_s, y_size_s)

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

    #print('Constructing conversion matrix y-edge')
    #y_sift_matrix  = make_conv_matrix(x_size_s, y_size_s, conv_y)
    #print('Constructing conversion matrix x-edge')
    #x_sift_matrix  = make_conv_matrix(x_size_s, y_size_s, conv_x)
    #print('Constructing conversion matrix yx-edge')
    #yx_sift_matrix = make_conv_matrix(x_size_s, y_size_s, conv_yx)
    #print('Constructing conversion matrix xy-edge')
    #xy_sift_matrix = make_conv_matrix(x_size_s, y_size_s, conv_xy)

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

    #print('Constructing blurring conversion matrix')
    #blurr_matrix = make_conv_matrix(x_size_s, y_size_s, conv_blurr)
    # ぼかしたのを引くと、ピークが残る。という方法でエッジのピークを抽出
    # SciPy に、何故か単位行列生成のメソッドが無い・・・

    #print('Constructing color conversion matrix to white and black')
    conv_WB = np.array([[1,1,1],\
                        [1,1,1],\
                        [1,1,1]])/3.0
    #conv_WB_matrix = make_color_matrix(x_size_s, y_size_s, conv_WB)

    img_last = cv2.resize(img, (x_size_s, y_size_s))
    ceiling_value = 250

    idnet = ids.ID_NetWork()
    project_ID = idnet.register_ID({"data_type": "project"}) # 1
    app_ID =     idnet.register_ID({"data_type": "app"})     # 2
    work_ID =    idnet.register_ID({"data_type": "work"})    # 3
    obj = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID} # なんでこれがいるのかわかんなくなってきた・・・
    repeat = True
    while repeat:
        if image != 'picture':
            ret, img = cam.read() # ret, img = にしないと動かない
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        #img_array = imgs.reshape([x_size_s*y_size_s*3])/256.0 # 1次元配列に変換。ここで２５６分の１に
        #img_xdiff = (np.array(x_sift_matrix.dot(img_array)))**2 # 行列による変換の核心部分
        #img_ydiff = (np.array(y_sift_matrix.dot(img_array)))**2
        #img_xydiff = (np.array(xy_sift_matrix.dot(img_array)))**2
        #img_yxdiff = (np.array(yx_sift_matrix.dot(img_array)))**2
        #img_diff_array = (img_ydiff + img_xdiff + img_xydiff + img_yxdiff)/reduction
        #img_diff_array = np.array(conv_WB_matrix.dot(img_diff_array)) # 検出したエッジイメージを白黒にする
        #img_diff_a_peak = np.maximum(0,img_diff_array - np.array(blurr_matrix.dot(img_diff_array)))*7.0 #ぼかしたのを引くと、ピークが残る
        #img_diff = img_diff_array.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        #img_diff_peak = img_diff_a_peak.reshape([y_size_s, x_size_s, 3]) # 元の配列の形に戻す。
        #_thre, img_diff2 = cv2.threshold(img_diff_peak, sh, 255, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(imgs, 100, 200) # opencv 組み込み関数によるエッジ検出
        
        # 漫画風変換。画像データを実数にした後は、２５０分の１にして返さないといけない？？？
        #img_draw = np.maximum(0, 0.1 + 0.5*np.maximum(0,imgs)/256.0 - np.maximum(0, img_diff2))
        cv2.imshow(filen, imgs)
        #cv2.imshow('diff', img_diff)
        #cv2.imshow('diff2', img_diff2)
        #cv2.imshow('draw', img_draw)
        #cv2.imshow('peak', img_diff_peak)
        cv2.imshow('Canny', img_canny)
        key = cv2.waitKey(1) # waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec) 
        if key == 27: # when ESC key is pressed break
            repeat = False

        #img_draw = np.array(np.minimum(255, np.maximum(0, 256*img_draw)),dtype='uint8')
        if image == 'picture':
            #cv2.imwrite('record_picture_diff.jpg',  img_diff)
            #cv2.imwrite('record_picture_diff2.jpg', img_diff2)
            #cv2.imwrite('record_picture_peak.jpg',  img_diff_peak)
            #cv2.imwrite('record_picture_draw.jpg',  img_draw)
            cv2.imwrite('record_picture_canny.jpg',  img_canny)
        else: # 動画
            # 時間微分用
            # 整数のままでハンドリングできる場合は、２５６分の１は要らない
            img_diff_v = abs(np.array(imgs,dtype='int16') - np.array(img_last,dtype='int16')) - sht
            img_diff_c = np.minimum(np.maximum(img_diff_v*amp, 0), ceiling_value)
            img_diff_t = np.array(img_diff_c,dtype='uint8')
            cv2.imshow('diff_time', img_diff_t)
            img_last = copy.copy(imgs)

            # 画像を1フレーム分として書き込み
            #writer.write(img_draw)
        # エッジ分割
        # 自作エッジをやめ、Canny によるエッジ img_canny に切り替えた
        # cv2.connectedComponentsWithStats によるエッジの分割を行う
        # https://axa.biopapyrus.jp/ia/opencv/object-detection.html
        #img_H, img_S, img_V = cv2.split(img_diff2) # ここに img_canny を入れるとおかしくなる。img_cannyが元々白黒フォーマットのため
        #img_H = np.array(img_H,dtype='uint8')
        _thre, img_flowers = cv2.threshold(img_canny, 140, 255, cv2.THRESH_BINARY) # img_H
        #nlabels, labelimages = cv2.connectedComponents(img_flowers)
        nlabels, labelimages, stats, centroids = cv2.connectedComponentsWithStats(img_flowers)
        #print('centroids = ', centroids) # centroidsは、分離した輪郭の重心位置(x,y)のリスト
        #print('nlabels',nlabels)
        #print('labelimages',labelimages)
        #height, width = img_cont.shape[0:2]
        cols = []
        # background is label=0, objects are started from 1 
        # ランダムなカラー生成
        for i in range(1, nlabels):
            cols.append(np.array([random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]))
        # 分割したエッジを別々の画像ファイルとして生成、img_contsにリストする
        # 色分けした画像 img_cont も作成
        img_ = copy.copy(imgs)
        img_black = np.zeros(img_.shape[0:3],dtype='uint8')
        img_cont = copy.copy(img_black)
        img_conts=[]
        detected_centroids = []
        for i in range(1, nlabels):
            if stats[i][4] >= 10:
                img_contn = copy.copy(img_black)
                img_cont[labelimages == i, ] = cols[i - 1]
                img_contn[labelimages == i, ] = cols[i - 1]
                img_conts.append(img_contn)
                detected_centroids.append(centroids[i])
                #cv2.imshow('contors', img_contn)
                #key = cv2.waitKey(0)
        cv2.imshow('contors', img_cont)
        key = cv2.waitKey(1)
        #print('Number of detected contors = ', len(img_conts))
        #print(img_cont)
        #break
        if image == 'picture':
            f = open('point_cloud.txt', 'w')
        # 輪郭を 点群に point cloud
        #new_clouds = []
        cont_cloud = pcl.PointCloud(2, xs=detected_centroids) # cv2で分離したエッジの塊を一つの点とした点群
        cont_cloud.ID = idnet.register_ID({"data_type": "point_cloud_small_num"}) # 少数点群なので、個々の点にＩＤつける筈だが、やってない
        # conts_as_cloud  # cont_cloud の各点に相当する点群
        indi_cont_clouds = []
        # indi_cont_clouds_top # conts_as_cloud を自作ソフトで分岐の無い線の群に分割、さらにソートしたトップ
        # indi_cont_clouds_disc # conts_as_cloud を自作ソフトで分岐の無い線の群に分割、さらにソートして捨てられたモノ
        for img in img_conts: # cv2で分離したエッジの塊が、一つのimgの中に入ってる
            # scan_contor関数は、y軸を反転しているので注意。
            coordi_list, data_list = scan_contor(x_size_s, y_size_s, img, imgs, cont_color=True) #imgs には特に意味は無い
            #print('Number of detected points = ', len(data_list))
            ###### 自作の点群ソーティングと分割
            #print('Sorting points and creating contols as point clouds start.')
            conts_as_cloud = pcl.PointCloud(2, xs=coordi_list)
            conts_as_cloud.points_line_sort()
            # ID と関係つけ
            conts_as_cloud.ID = idnet.register_ID({"data_type": "point_cloud_large_num"})
            coment =  idnet.register_relation( fromID=conts_as_cloud.ID, toID=cont_cloud.ID, strength=0.5, relation="part", relation_property_dat=obj)
            #print('Dividing contors start.')
            indi_cont_clouds_loco = conts_as_cloud.cloud_division(far_dist=2) #
            for cloud in indi_cont_clouds_loco:
                cloud.ID = idnet.register_ID({"data_type": "point_cloud_large_num"})
                coment =  idnet.register_relation( fromID=cloud.ID, toID=conts_as_cloud.ID, strength=0.5, relation="part", relation_property_dat=obj)
            #print('Number of contors = ', len(indi_cont_clouds_loco))
            indi_cont_clouds += indi_cont_clouds_loco
            #print('Extracting long contols.')
        indi_cont_clouds_top, indi_cont_clouds_disc = sort_group_clouds(indi_cont_clouds)
        #print('Number of extracted long contors was =', len(indi_cont_clouds_top))
        img_cont2 = create_colored_contors(x_size_s, y_size_s, img_black, indi_cont_clouds_top, indi_cont_clouds_disc)
        cv2.imshow('contors_color', img_cont2)
        if image == 'picture':
            cv2.imwrite('record_picture_cont2.jpg',  img_cont2)
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            repeat = False
        ### 参照点群とのマッチング評価を行う pict2point3.py ではこの機能は試さない ###
        #point_matching(coordi_list_ref, coordi_list)
        #cv2.imwrite('record_picture_cont.jpg', img_cont)
        #### 抽出した線の評価を行う ####
        k = 360.0/blurred_rate*5.0
        frequency_list =  [2./k,   3./k,   4./k,   5./k,   6./k,  8./k,  10./k, 12.5/k, 16.0/k, 20./k, 25./k, 30./k,  40./k,  50./k, 60./k]
        m = 1.5/blurred_rate*5.0
        decay_time_list = [250.*m, 167.*m, 125.*m, 100.*m, 83.*m, 62.*m, 50.*m, 40.*m,  31.*m,  25.*m, 20.*m, 16.7*m, 12.5*m, 10.*m, 8.3*m]
        fouriers = []
        for cloud in indi_cont_clouds_top:
            #print('ID=, ', cloud.ID, file=f)
            #for x in cloud.xs:
                #print(x[0], ', ', x[1], file=f)
            #print('\n\n', file=f)
            fl = frt.Fourier(frequency_list, decay_time_list, time_new=0.)
            fouriers.append(fl)
            x_last = cloud.xs[0]
            dist_total = 0.0
            vect_x = 0.0
            vect_y = 0.0
            ratio = 0.3*blurred_rate/5.0 #0.3
            # フーリエ成分評価開始 各点群に対して行う
            for x in cloud.xs:
                dist_x = x[0] - x_last[0]
                dist_y = x[1] - x_last[1]
                vect_x = (1-ratio)*vect_x + dist_x*ratio
                vect_y = (1-ratio)*vect_y + dist_y*ratio
                dist_total += math.sqrt(dist_x**2 + dist_y**2)
                side_wind = vect_x*dist_y - vect_y*dist_x
                fl.fourier_trans(imput=side_wind, time_now=dist_total)
                x_last = copy.copy(x)
            # フーリエ変換終了
            #print('ID = ', cloud.ID)
            #print('posision = ', cloud.centre[0], cloud.centre[1])
            # ギザギザ検出に特化して、評価関数を作った。point_cloud.xlsx参照
            compo_total = 0.0
            for compo in fl.components:
                compo_total += (compo["Real_sqr_peak"] + compo["Imagin_sqr_peak"])/(compo['frequency']*k)*10.
                #print("Frequency = ",compo["frequency"]*360.)
                #print("Real    = ",compo["Real"])
                #print("Imagin  = ",compo["Imagin"])
                #print("Residue = ",compo["Residue"])
                #print("Ref_sqr = ",compo["Ref_sqr"])
                #print("Ref_abs = ",compo["Ref_abs"])
                #print("Residue_removed = ", component)
                #print("next")
            compo_total += 0.000000000000000001
            gizagiza = 3.3*((fl.components[6]['Real_sqr_peak'] + fl.components[6]['Imagin_sqr_peak'])/(fl.components[6]['frequency']*k)*10 + \
                        (fl.components[7]['Real_sqr_peak'] + fl.components[7]['Imagin_sqr_peak'])/(fl.components[7]['frequency']*k)*10 + \
                        (fl.components[8]['Real_sqr_peak'] + fl.components[8]['Imagin_sqr_peak'])/(fl.components[8]['frequency']*k)*10 \
                        )/compo_total
            if gizagiza > 1.0:
                textcolor = (10,250,250) # yellow
                if image == 'picture':
                    print('GIZAGIZA ID =', cloud.ID, 'posision = ', cloud.centre[0], cloud.centre[1], file=f)
                    for compo in fl.components:
                        print("Frequency = ",compo["frequency"]*k, file=f)
                        print("Real_sqr_peak   = ",compo["Real_sqr_peak"], file=f)
                        print("Imagin_sqr_peak = ",compo["Imagin_sqr_peak"], file=f)
                cv2.putText(img_cont2, "giza", (int(cloud.centre[0])-8, y_size_s - int(cloud.centre[1])), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, textcolor, 1, cv2.LINE_AA)
            funyafunya = 6.*((fl.components[3]['Real_sqr_peak'] + fl.components[3]['Imagin_sqr_peak'])/(fl.components[3]['frequency']*k)*10 + \
                            (fl.components[4]['Real_sqr_peak'] + fl.components[4]['Imagin_sqr_peak'])/(fl.components[4]['frequency']*k)*10  \
                            )/compo_total
            if funyafunya > 1.0:
                textcolor = (200,250,250) # pink
                if image == 'picture':
                    print('FUNYAFUNYA ID =', cloud.ID, 'posision = ', cloud.centre[0], cloud.centre[1], file=f)
                    for compo in fl.components:
                        print("Frequency = ",compo["frequency"]*k, file=f)
                        print("Real_sqr_peak   = ",compo["Real_sqr_peak"], file=f)
                        print("Imagin_sqr_peak = ",compo["Imagin_sqr_peak"], file=f)
                cv2.putText(img_cont2, "Funya", (int(cloud.centre[0])-8, y_size_s - int(cloud.centre[1])), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, textcolor, 1, cv2.LINE_AA)
        
        cv2.imshow('contors_color_gi', img_cont2)
        if image == 'picture':
            wait=0
            cv2.imwrite('record_picture_cont2_fu.jpg',  img_cont2)
            repeat = False
        else:
            wait=1
            writer.write(imgs)
            writer2.write(img_cont2)

        key = cv2.waitKey(wait)# waitkey を入れないと画像が更新しないらしい。引数は待ち時間(msec)
        if key == 27: # when ESC key is pressed break
            repeat = False
           
    if image == 'picture':
        f.close()
    else:
        writer.release() # ファイルを閉じる
        writer2.release()
        cam.release()
    cv2.destroyAllWindows()
