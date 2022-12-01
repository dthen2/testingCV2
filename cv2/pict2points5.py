import copy
import math
import cv2
import numpy as np
import random
#import tensorflow as tf
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from matplotlib import pylab as plt
import point_cloud7 as pcl

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

# pict2points2.py では、輪郭をCannyで抽出後、点群として抽出し、参照画像とのマッチング比較を行った。左記マッチングとは別途、opencvによる分割も試した
# pict2points5.py では、pict2points.py をベースに再び、点群として抽出して参照画像とのマッチング比較を行うものを制作。point_cloud7を使用

# 点群への変換
# 抽出した輪郭（img_draw の非ゼロ）の座標と、強度（img_diff）をリストにして返す
# img_diff, img_diff はnumpy配列である事。
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

# 点群のみを返すシンプル版
def scan_contor_points(x_size_s, y_size_s, img_cont, cont_color=True):
    coordi_list = []
    #data_list = []
    for y in range(y_size_s):
        for x in range(x_size_s):
            #row = y*x_size_s*3 + x*3
            if cont_color:
                point = img_cont[y, x, 1]
            else:
                point = img_cont[y,x]
            if point: 
                coordi_list.append([float(x), float(y_size_s - y)])
                #data_list.append(float(img_diff[y, x, 0]))
    return coordi_list #, data_list


def plotdata(ax,closest, inv=False):
    if inv:
        centre_ref = np.array(closest.size_ref["centre"])
        transinv = np.linalg.inv(closest.transform_best)
        xsx = []
        xsy = []
        for xs in closest.xs:
            xsx.append(xs[0])
            xsy.append(xs[1])
        ax.scatter(xsx,xsy, c="black", s=5.0, marker='o')
        xrx = []
        xry = []
        for ref in closest.ref:
            refnp = np.array(ref)
            xr = np.dot(refnp - centre_ref,transinv) - closest.sift_best
            xrx.append(xr[0])
            xry.append(xr[1])
        ax.scatter(xrx,xry, c="red", s=5.0, marker='o')
    else:
        xsd=closest.xs
        refp=closest.ref
        sift = closest.sift_best
        centre_ref = closest.size_ref["centre"]
        xsx = []
        xsy = []
        for i in range(len(xsd)):
            xsr = np.dot(np.array(xsd[i]) + sift, closest.transform_best)
            xsx.append(xsr[0])
            xsy.append(xsr[1])
        ax.scatter(xsx,xsy, c="black", s=5.0, marker='o')
        xrx = []
        xry = []
        for i in range(len(refp)):
            xr = refp[i][0] - centre_ref[0]
            yr = refp[i][1] - centre_ref[1]
            xrx.append(xr)
            xry.append(yr)
        ax.scatter(xrx,xry, c="red", s=5.0, marker='o')


##########################################################################
if __name__ == '__main__':

    #image = 'jetson_nano_web_cam' # 今持ってるUSBカメラは、2回目のrunで照度が下がる？？？
    #image = 'normal_cam'
    #image = 'file'
    image = 'picture'
    pict_file_name = 'nana.jpg'
    pict_file_name = 'IMG_3194.jpg' # 手に持ったスパナ
    pict_file_name = 'IMG_3195.jpg' # 手に持ったスパナ
    #pict_file_name = 'IMG_3199_.jpg' # フカダがスパナを持つ画像
    #pict_file_name = 'IMG_4368.jpg'

    #pict_ref_file_name = 'spanner2.jpg'
    pict_ref_file_name = 'IMG_4368.jpg' # これもスパナ

    inv = False # 参照とデータを入れ替える場合はTrue。False が正常

    blurred_rate = 10.0 # データ画像の縮小率
    blurred_ref = 10.0  # 参照画像の縮小率
    
    canny_th1 = 100#300     # 輪郭抽出閾値1（線の隣は閾値を下げる処理をしている）
    canny_th2 = 200#400     # 輪郭抽出閾値2
    canny_th1_ref = 150     # 輪郭抽出閾値（線の隣は閾値を下げる処理をしている）
    canny_th2_ref = 250     # 輪郭抽出閾値

    resolution_rand = 10.5    # 乱数探索時のresolution
    MAX_MAGNIFICATION = 5.0

    if image == 'picture': # 静止画
        # https://peaceandhilightandpython.hatenablog.com/entry/2015/12/23/214840
        filen = 'picture_file'
        img = cv2.imread(pict_file_name) # .画像ファイル
        img_ref = cv2.imread(pict_ref_file_name)
        #sh = 3.0 # 2.0
        y_size_r, x_size_r, cc = img_ref.shape
        x_size_sr = int(x_size_r/blurred_ref)
        y_size_sr = int(y_size_r/blurred_ref)
        # 参照データとの比較開始
        imgs_ref = cv2.resize(img_ref, (x_size_sr, y_size_sr))
        img_canny_ref = cv2.Canny(imgs_ref, canny_th1_ref, canny_th2_ref) #150,250
        cv2.imshow(pict_ref_file_name, img_canny_ref)
        key = cv2.waitKey(1)
        #coordi_list_ref, data_list_ref = scan_contor(x_size_sr, y_size_sr, img_canny_ref, imgs_ref,cont_color=False)
        coordi_list_ref = scan_contor_points(x_size_sr, y_size_sr, img_canny_ref, cont_color=False)
    
        y_size, x_size, c = img.shape
        print('Picture size =', x_size, y_size)
        x_size_s = int(x_size/blurred_rate)
        y_size_s = int(y_size/blurred_rate)
        imgs = cv2.resize(img, (x_size_s, y_size_s))
        print('Reduced picture size =', x_size_s, y_size_s)

        img_canny = cv2.Canny(imgs, canny_th1, canny_th2) # opencv 組み込み関数によるエッジ検出
        cv2.imshow(filen, imgs)
        cv2.imshow('Canny', img_canny)
        cv2.imwrite('record_picture_canny.jpg',  img_canny)

        # 輪郭を 点群に point cloud
        print('Detecting contors start.')
        #coordi_list, data_list = scan_contor(x_size_s, y_size_s, img_canny, img_diff, cont_color=False)
        coordi_list = scan_contor_points(x_size_s, y_size_s, img_canny, cont_color=False)
        print('Number of detected points = ', len(coordi_list))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fig.show()
        if inv:
            plotinv=False
            closest = pcl.Closest(2, coordi_list_ref, coordi_list)
        else:   # こちらが正常（refに部品のデータ）
            plotinv=True
            closest = pcl.Closest(2, coordi_list, coordi_list_ref) 
        closest.MAX_MAGNIFICATION = MAX_MAGNIFICATION
        closest.RATE_UPDATE_SCALE = 0.4
        mat = 0.0
        fig.show()
        closest.resolution = resolution_rand
        closest.MAX_MAGNIFICATION = 3.
        mat_prev = 0.
        print('Point matching started.')
        # 乱数による探索
        for i in range(20):
            #plt.cla()
            #plotdata(ax,closest)
            #fig.show()
            #plt.pause(0.01) #これが無いと何故か表示されない
            mat = closest.closest_point_search( search_num=20, matching_func_best=mat, transmode='mag')
            if mat > mat_prev:
                plt.cla()
                plotdata(ax,closest, inv=plotinv)
                #plotdata_anti(ax,closest) # refの方を座標変換
                fig.show()
                plt.pause(0.001) #これが無いと何故か表示されない
                print('at i=%d, matching = %f'%(i, mat))
            mat_prev = mat
            #print('mat=',mat)
    
        # ここから、グラジェントに基づいて行う
        closest.resolution = 2.5 # 1.5 分解能変更
        # ここから、グラジェントに基づいて探索
        # 点間隔をcloud.dist_mean として評価するためにPointCloudを使う。点間隔がgivenならこの演算は不要
        cloud = pcl.PointCloud(2, xs=coordi_list, points=[])
        cloud.points_line_sort()
        closest.trans_best_to_default()
        print('dist_mean = %.4f' % cloud.dist_mean)
        #print(coordi_list)
        x_scale   = cloud.dist_mean*2.
        mag_scale = cloud.dist_mean/cloud.cloud_radius*20.
        converge_num, mat, is_converge_sift, is_converge_mag = closest.closest_gradient_approach(x_scale, mag_scale, number=50, tolerance=0.0)
        plt.cla()
        plotdata(ax,closest, inv=plotinv)
        fig.show()
        plt.pause(0.01) #これが無いと何故か表示されない
        print('n=%d, mat=%f, converge='%(converge_num+1, mat), is_converge_sift, is_converge_mag)
        print('Ended')
        plt.show()
           
    #if image == 'picture':
    #    cv2.waitKey(1)
    #cv2.destroyAllWindows()
