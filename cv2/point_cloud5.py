# 点群データの扱い
# AIノートの21/March/2021以降の考察に基づく
# 新たに、座標変換による類似度評価を新たなクラスとするもの。作成途上
# 2021Oct21 刷新してVer.5
from matplotlib import pylab as plt
import numpy as np
import math
import random
import copy
import time

#MAX_DIMENSION = 5
MAX_SCALE = 10000000000.0
MIN_SCALE = -10000000000.0

def create_dimensional_list(dimension, value):
    lis = []
    for n in range(dimension):
        lis.append(value)
    return lis

def cloud_centre(xs, dimension):
    # 点群の中心位置を評価する
    x_sum  = create_dimensional_list(dimension, 0.0)
    centre = create_dimensional_list(dimension, 0.0)
    for x in xs:
        for n in range(dimension):
            x_sum[n] += x[n]
    for n in range(dimension):
        centre[n] = x_sum[n]/len(xs)  # 点群の中心位置
    return centre

# point_F クラスを定義
# 単独の点のデータフォーマット
# 30/April/2021 の定義に従う
# 5次元空間まで扱う → 見直して、無制限に
# face は「顔の向いてる方向」のベクトル
# 見直し → # ポイントクラウド（点群）は、単に点のリスト
# 見直し → # 並べ替えや再グループ化を自由に行うためには、クラウドは単にリストの方が良い
# 見直し 自由に使えるフラグを廃止し、Cloudに移した 2021Oct21
class Point_F:
    def __init__(self, dimension):
        self.dim = dimension    # 次元。デフォルトは3次元。現状、使ってない（点群に対して付与している）
        self.ID = 0             # デフォルトのゼロは「ＩＤ無し」を意味する。
        self.x     = create_dimensional_list(dimension, 0.0) # 座標。[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
        self.v     = create_dimensional_list(dimension, 0.0) # 速度
        self.face1 = create_dimensional_list(dimension, 0.0) # 方向
        self.strength = 0.0    # 強度
        self.radius   = 0.0    # テリトリーとも言う、半径のこと
        self.face1_radius = 0.0    
        # face1_radius は、face方向の半径を言う。radius より大きい値なら、face方向を向いた細長い物体、小さい値ならface方向を向いた平べったい物体
        self.direction = True    # 方向の有無。異方性はあっても方向はない場合があるので、このフラグは必要 
        self.group_ID = 0    # 属するグループのID

############################################
# 点群のクラス。名称とコード上の位置変更。
# 仕様変更し、単純に点の座標リストと、Point_Fのリストのどちらも受け入れる。
# つまり、少数点群にも多数点群にも対応できる
# points は、 Point_F のリスト
# x_list は、[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
# 両方が入力されちゃったら、pointsが優先されるので注意
class PointCloud:
    def __init__(self, dimension, xs=[], points=[]):
        self.ID = 0 # デフォルトのゼロは「ＩＤ無し」を意味する。
        self.points = points  # 点群本体（Point_Fのリスト）
        if points:
            self.xs = []
            for point in points:
                self.xs.append(point.x)
        else:
            self.xs = xs
        self.dim = dimension    # 次元
        self.len = len(self.xs)   # 点の数
        self.centre = create_dimensional_list(dimension, 0.0)
        self.vector_cloud = []  # points_line_sort() を実施して書き込まれる。点間相対位置ベクトル。点群より一つ数が減るので別のリストとする。別ルーチンで書き込まれる
        self.distance = []      # 点間相対距離。点群より一つ数が減るので別のリストとする。別ルーチンで上記vector_cloudと共に書き込まれる
        self.dist_mean = 0.0    # 上記点間相対距離の平均値。一筆書きにソートしないと評価できないので、上記別ルーチンで書き込まれる
        # 点群の中心位置を評価する
        self.centre = cloud_centre(self.xs, dimension)
        #x_sum = create_dimensional_list(dimension, 0.0)
        #for x in self.xs:
        #    for n in range(dimension):
        #        x_sum[n] += x[n]
        #for n in range(dimension):
        #    self.centre[n] = x_sum[n]/self.len  # 点群の中心位置
        # 点群の大雑把な大きさと、中心から最も遠い点を特定する
        max_radius = 0.0
        radius_total = 0.0
        far_most_num = 0    # 点の数が１個だった場合は必要
        for num in range(self.len):
            radius = 0.0
            for n in range(dimension):
                radius += (self.xs[num][n]-self.centre[n])**2
            radius_total += radius
            if radius > max_radius:
                far_most_num = num
                max_radius = radius
        self.far_most_num = far_most_num    # 中心から最も遠い点の番号
        self.cloud_radius = radius_total/self.len   # 点群の大雑把な半径
        self.flag1 = create_dimensional_list(self.len, False)  # 自由に使えるフラグ
        self.flag2 = create_dimensional_list(self.len, False)  # 自由に使えるフラグ

    # 中心位置や中心から最も遠い点を評価
    # 点群生成時に既に評価しているが、点群を分割・合成した際に再評価するためにある
    def centre_and_far_most(self):
        # 点群の中心位置を評価する
        self.centre = cloud_centre(self.xs, self.dim)
        #x_sum = create_dimensional_list(self.dim, 0.0)
        #for x in self.xs:
        #    for n in range(self.dim):
        #        x_sum[n] += x[n]
        #for n in range(self.dim):
        #    self.centre[n] = x_sum[n]/self.len  # 点群の中心位置
        
        # 点群の大雑把な大きさと、中心から最も遠い点を特定する
        max_radius = 0.0
        radius_total = 0.0
        far_most_num = 0    # 点の数が１個だった場合は必要
        #print('xs', self.xs)
        #print('len', self.len)
        for num in range(self.len):
            radius = 0.0
            for n in range(self.dim):
                radius += (self.xs[num][n]-self.centre[n])**2
                #print(num)
            radius_total += radius
            if radius > max_radius:
                far_most_num = num
                max_radius = radius
        self.far_most_num = far_most_num    # 中心から最も遠い点の番号
        self.cloud_radius = radius_total/self.len   # 点群の大雑把な半径

    def resetflags(self):
        self.flag1 = create_dimensional_list(self.len, False)
        self.flag2 = create_dimensional_list(self.len, False)

    # 点群を、近い順に並べ替えて一筆書きにする
    # 8/Jul/2021 のノート参照
    def points_line_sort(self):
        self.vector_cloud = []  # ベクトル群はリセットする
        self.distance = []  # 距離群もリセットする
        if self.points:
            new_points = [self.points[self.far_most_num]] 
        new_xs = [self.xs[self.far_most_num]]   # 並べ直した点群の１番目は、最も中央から遠い点
        vector = create_dimensional_list(self.dim, 0.0)
        dist_total = 0.0
        self.resetflags()
        i = self.far_most_num   # 最も中央から遠い点から始める
        # 一番目は既にリストに入れているので、len-1 回まわす
        for dummy in range(self.len-1): # range()はゼロから始まり、end値の一つ前でまで出す。リストもゼロ番から始まる
            closest_dist = MAX_SCALE
            for j in range(self.len): # クラウド内をスキャン 
                if i != j and self.flag1[j]==0:
                    dist = 0.0
                    vector_correction = 0.0
                    vector_corr_abs = 0.0
                    for n in range(self.dim):
                        vector[n] = self.xs[j][n]-self.xs[i][n]
                        dist += vector[n]**2
                        if self.dist_mean > 0.0 and i != self.far_most_num:
                            vector_correction += self.vector_cloud[-1][n]*vector[n]
                            vector_corr_abs += self.vector_cloud[-1][n]**2
                    if self.dist_mean > 0.0 and i != self.far_most_num:
                        dist_corrected = dist*(1.0 - vector_correction/vector_corr_abs*self.dist_mean/max(dist, self.dist_mean))
                    else :
                        dist_corrected = dist  # 既にソートができている場合、ベクトルに基づく補正を行う
                        # ベクトルに基づく補正とは、前回までの点間ベクトルの方向にある点の方を近いと判定すること
                    if dist_corrected < closest_dist: # 近ければ更新
                        closest_dist = dist
                        j_near = j
            self.vector_cloud.append(vector)
            self.distance.append(closest_dist)
            dist_total += closest_dist
            if self.points:
                new_points.append(self.points[j_near])
            new_xs.append(self.xs[j_near])
            self.flag1[i] = 1
            self.flag1[j_near] = 1
            i = j_near
        #new_points.append(self.points[j_near])    # 一番最後の点
        if self.points:
            self.points = new_points # ソートした点群に置き換え
        self.xs = new_xs  # ソートした点群に置き換え
        self.far_most_num = 0   # 最も遠い点は１番目になる
        self.dist_mean = dist_total/self.len


    # 点群列の切れ目を認識して分割する
    # 上記point_line_sort関数で一筆書き化とベクトル群が得られている事が条件
    def cloud_division(self, far_dist=0):
        if far_dist == 0:
            far_dist = self.dist_mean + 0.150*np.std(self.distance)    # 「遠い」とする判定基準。点間距離の平均値＋距離分布の３σ
        #print(self.dist_mean)
        #print(np.std(self.distance))
        edges = []
        iterate = 0
        for dist in self.distance:
            #print(dist)
            #print(iterate)
            if dist > far_dist: # 遠く離れてる所があれば、切れ目と認識
                self.flag2 = 1
                edges.append(iterate)
            iterate += 1
        new_clouds = []
        prev = 0
        #print('xs_all', self.xs)
        #print('edges', edges)
        #print('len', len(self.points))
        for edge in edges:  # 点群を分割する
            #print('edge', edge)
            #print('prev', prev)
            new_cloud = copy.copy(self) # パラメータ類を継承するために先ず複製
            if self.points:
                new_cloud.points = self.points[prev:edge+1]
            else:
                new_cloud.points = []
            new_cloud.xs = self.xs[prev:edge+1]
            new_cloud.vector_cloud = self.vector_cloud[prev:edge]
            new_cloud.distance = self.distance[prev:edge]
            new_cloud.len = edge + 1 - prev
            new_cloud.centre_and_far_most() # 中心位置などを再評価 print
            new_clouds.append(new_cloud)    # 分割した点群を点群のリストとして納める
            prev = edge+1
        #print('div end')
        if edges:   # 分割した最後の部分
            new_cloud = copy.copy(self) # パラメータ類を継承するために先ず複製
            if self.points:
                new_cloud.points = self.points[prev:]
            else:
                new_cloud.points = []
            new_cloud.xs = self.xs[prev:]
            new_cloud.vector_cloud = self.vector_cloud[prev:]
            new_cloud.distance = self.distance[prev:]
            new_cloud.len = self.len - prev
            new_cloud.centre_and_far_most() # 中心位置などを再評価
            new_clouds.append(new_cloud)
        if new_clouds==[]:
            return [self]       # 分割が無ければそのまんま
        else:
            return new_clouds   # 分割した点群のリストを返す。


##################################################################################################
######### Closest class ##########################################################################
# 点群のサイズを評価して返すローカル関数
# 各座標の最大最小、中心値、大きさ（最大最小の差）
# xs 点群(座標のリスト)[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
def cloudSize(xs, dimension):
    max_x = create_dimensional_list(dimension, MIN_SCALE)
    min_x = create_dimensional_list(dimension, MAX_SCALE)
    centre = []
    scale = []
    for n in range(dimension):
        for x in xs:
            max_x[n] = max(max_x[n], x[n])
            min_x[n] = min(min_x[n], x[n])
        centre.append( (max_x[n] + min_x[n])/2)
        scale.append( max_x[n] - min_x[n])
    size = {"max": max_x, "min": min_x, "centre": centre, "scale": scale}
    return(size)

# 点群2つの一致度探索パラメータを納めたクラスを定義する
# xs 点群(座標のリスト)[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
# ref 比較対象の点群(座標のリスト)[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
# sift, magnify, rotation 入力された点群を変換するパラメータ。refに一致させることを目指す
# points の使用はやめ
class Closest:
    def __init__(self, dimension, xs, ref ):
        self.xs = xs
        self.ref = ref
        self.size_cloud = cloudSize(xs,  dimension)
        self.size_ref   = cloudSize(ref, dimension)
        sift = []
        magnify = []
        rotation = []
        scale_cloud = 0.0
        scale_ref = 0.0
        for n in range(dimension):
            scale_cloud += self.size_cloud["scale"][n]
            scale_ref += self.size_ref["scale"][n]
        scale_fact = scale_ref/scale_cloud
        for n in range(dimension):
            sift.append(- self.size_cloud["centre"][n])
            magnify.append(scale_fact)
            rotation.append( 0.0 )
        self.scale_fact = scale_fact    # 大雑把なスケール差のパラメータ
        self.sift = sift
        self.magnify = magnify
        self.rotation = rotation
        self.dimension = dimension
        self.sift_best = copy.copy(self.sift)
        self.magnify_best = copy.copy(self.magnify)
        self.rotation_best = copy.copy(self.rotation)
        # 部分一致をここでやると、おかしなことになるので、MAX_MAGNIFICATION の値は控えめにして、部分一致はグループの再構成で対応する事にする
        self.MAX_MAGNIFICATION = 2.0
        self.RATE_UPDATE_SCALE = 0.8

    # 評価関数本体。一致度が高い（point 同士の距離が近い）と、高い値を返す
    def matching_function(self, diff2, resolution):
        match = resolution**2 /(diff2 + resolution**2)
        return(match)

    def coordinate_transformation(self):
        transed_points = []
        for x in self.xs:
            sin = math.sin(self.rotation[0])
            cos = math.cos(self.rotation[0])

            # とりあえず、2次元の場合のみ作成
            new_x = create_dimensional_list(self.dimension, 0.0)
            new_x[0] = (x[0] + self.sift[0])*self.magnify[0]*cos - (x[1] + self.sift[1])*self.magnify[1]*sin
            new_x[1] = (x[1] + self.sift[1])*self.magnify[1]*cos + (x[0] + self.sift[0])*self.magnify[0]*sin
            transed_points.append(new_x)
        return transed_points

    # 特定の座標変換に対して、一致度を評価する
    # refferenceは変換しない
    # resolution のスケールは、拡大率の異方性を考慮せず、scale_factのみで補正する
    # 参照点群ref の各点の、face方向にデータ点群の点が入っていたら距離を減らす補正をして、参照点のつながりを表現する。
    #  この処理は2次元にしか対応しておらず、再考を要する。
    def closest_point_evaluation(self, resolution):
        transed_points = self.coordinate_transformation()
        matching_func_total = 0.0
        for new_x in transed_points:
            matching_func_point = 0.0
            for x_r in self.ref:
                diff_vector = create_dimensional_list(self.dimension, 0.0)
                diff2 = 0.0
                face_diff = 0.0
                for n in range(self.dimension):
                    diff_vector[n] = new_x[n] - x_r[n] + self.size_ref["centre"][n]
                    diff2 += diff_vector[n]**2
                    # face方向に入っていたら距離を減らす補正をして、点のつながりを表現する・・・廃止
                    #face_diff += (diff_vector[n])*(point_r.face1[n])
                diff2 /= max(1.0,face_diff/math.sqrt(diff2)) 
                matching_func_point += self.matching_function(diff2, resolution*self.scale_fact)
            matching_func_total += matching_func_point
        matching_func_norm = matching_func_total/max(len(self.xs), len(self.ref) )
        return(matching_func_norm)


    # 座標変換のパラメータをランダムに作成
    # rate_update = 1.0 で完全ランダム
    # rate_update = 0.0 で、変換を固定
    # メモ、random.random() は 0.0 から 1.0 の乱数 
    def transformation_random(self, rate_update):
        for n in range(self.dimension):
            sift_new = - self.size_cloud["centre"][n] + (random.random()-0.5)*self.scale_fact
            self.sift[n] = sift_new*rate_update + self.sift_best[n]*(1.0-rate_update)
            magnify_new = self.scale_fact*(self.MAX_MAGNIFICATION**(2*random.random()-1.0)) # メモ 「**」はべき乗
            self.magnify[n] = magnify_new*rate_update + self.magnify_best[n]*(1.0-rate_update)
            rot_new =  (2*random.random()-1.0)*math.pi
            self.rotation[n] = rot_new*rate_update + self.rotation_best[n]*(1.0-rate_update)

    # 最善の変換条件を記録
    def trans_best(self):
        self.sift_best = copy.copy(self.sift)
        self.magnify_best = copy.copy(self.magnify)
        self.rotation_best = copy.copy(self.rotation)
    
    # 最善の変換条件を記録
    def trans_best_to_default(self):
        self.sift = copy.copy(self.sift_best)
        self.magnify = copy.copy(self.magnify_best)
        self.rotation = copy.copy(self.rotation_best)

    # 様々な座標変換を試し、一致度探索を一定回数行う
    # resolution のアップデートはここでは行わず、別途行う（問題の性質によって様々な応答がありうるため
    def closest_point_search(self, search_num, resolution, matching_func_best):
        # rate_update = 1.0 で完全ランダム
        # rate_update = 0.0 で、変換を固定
        # rate_update の決め方を、point_cloud2.pyから変えた
        rate_update = 1.0 
        for i in range(search_num):
            self.transformation_random(rate_update)
            matching_func_evaluation = self.closest_point_evaluation(resolution)
            if matching_func_evaluation > matching_func_best:
                # 良い一致が得られたら、座標変換パラメータの変化を少なくする
                rate_update *= max(self.RATE_UPDATE_SCALE, matching_func_best/matching_func_evaluation)
                matching_func_best = matching_func_evaluation
                self.trans_best()
        return(matching_func_best)

    # 各変換に対する微分係数を求める 2021Oct21 追加
    def closest_point_diff(self, resolution, rot_scale, x_scale, mag_scale):
        rot_prev = copy.copy(self.rotation)
        sift_prev = copy.copy(self.sift)
        mag_prev = copy.copy(self.magnify)
        diff_rot = []
        diff_x = []
        diff_mag = []
        matching_func_norm = self.closest_point_evaluation(resolution)
        for n in range(self.dimension):
            self.rotation[n] += rot_scale
            matching_func = self.closest_point_evaluation(resolution)
            diff_rot.append((matching_func - matching_func_norm)/rot_scale)
            self.rotation[n] = rot_prev[n]

            self.sift[n] += x_scale
            matching_func = self.closest_point_evaluation(resolution)
            diff_x.append((matching_func - matching_func_norm)/x_scale)
            self.sift[n] = sift_prev[n]

            self.magnify[n] += mag_scale
            matching_func = self.closest_point_evaluation(resolution)
            diff_mag.append((matching_func - matching_func_norm)/mag_scale)
            self.magnify[n] = mag_prev[n]

        return diff_rot, diff_x, diff_mag

    def closest_point_approach(self, resolution, matching_func_best, diff_x, diff_rot, diff_mag):
        for n in range(self.dimension):
            self.sift[n] += diff_x[n] 
            self.magnify[n] += diff_mag[n] 
            self.rotation[n] += diff_rot[n]  

        matching_func_evaluation = self.closest_point_evaluation(resolution)
        if matching_func_evaluation > matching_func_best:
            matching_func_best = matching_func_evaluation
            is_converge = False
            self.trans_best()
        else:
            is_converge = True
            #self.trans_best() ######
            return matching_func_best, is_converge
        return matching_func_best, is_converge


########## Closest class 終わり ############


###############################
# パターンマッチング試験データ

p1  = [15.0, 13.0]
p2  = [13.0, 7.0]
p3  = [11.0, 3.0]
p4  = [ 6.0, 1.0]
p5  = [ 2.5, 6.0]
p6  = [ 5.0, 11.0]
p7  = [10.0, 16.0]
p8  = [18.0, 17.0]
p9  = [23.0, 12.0]
p10 = [22.0, 4.0]
p11 = [23.0, 8.0]
p12 = [21.0, 15.0]
p13 = [14.0, 17.0]
p14 = [ 7.5, 14.0]
p15 = [ 3.5, 9.0]
p16 = [ 2.3, 3.5]
p17 = [ 3.3, 1.2]
p18 = [ 8.5, 1.7]
p19 = [12.0, 5.3]
p20 = [14.3, 10.0]
xsdata = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]

r1 = Point_F(2)
r1.x = [13.6, 16.0]
r1.face1 = [0.0, 0.0]
r2 = Point_F(2)
r2.x = [13.8, 13.0]
r2.face1 = [-0.1, 1.0]
r3 = Point_F(2)
r3.x = [13.0, 9.0]
r3.face1 = [0.2, 1.0]
r4 = Point_F(2)
r4.x = [11.5, 5.0]
r4.face1 = [0.5, 1.0]
r5 = Point_F(2)
r5.x = [8.5, 2.0]
r5.face1 = [1.0, 0.7]
r6 = Point_F(2)
r6.x = [5.6, 1.9]
r6.face1 = [1.0, -0.5]
r7 = Point_F(2)
r7.x = [3.9, 4.7]
r7.face1 = [0.4, -1.0]
r8 = Point_F(2)
r8.x = [3.8, 9.0]
r8.face1 = [-0.1, -1.0]
r9 = Point_F(2)
r9.x = [5.0, 12.7]
r9.face1 = [-0.3, -1.0]
r10 = Point_F(2)
r10.x = [6.6, 15.9]
r10.face1 = [-0.6, -0.9]
r11 = Point_F(2)
r11.x = [10.0, 19.0]
r11.face1 = [-1.0, -0.6]
r12 = Point_F(2)
r12.x = [13.0, 20.7]
r12.face1 = [-1.0, -0.2]
r13 = Point_F(2)
r13.x = [17.0, 20.8]
r13.face1 = [-1.0, 0.2]
r14 = Point_F(2)
r14.x = [20.2, 19.2]
r14.face1 = [-0.5, 0.8]
r15 = Point_F(2)
r15.x = [22.3, 15.0]
r15.face1 = [-0.3, 1.0]
r16 = Point_F(2)
r16.x = [22.6, 11.0]
r16.face1 = [0.1, 1.0]
r17 = Point_F(2)
r17.x = [22.8, 13.2]
r17.face1 = [0.0, 0.0]
r18 = Point_F(2)
r18.x = [21.5, 17.5]
r18.face1 = [0.0, 0.0]
r19 = Point_F(2)
r19.x = [18.8, 20.2]
r19.face1 = [0.0, 0.0]
r20 = Point_F(2)
r20.x = [15.0, 21.0]
r20.face1 = [0.0, 0.0]
r21 = Point_F(2)
r21.x = [11.5, 20.0]
r21.face1 = [0.0, 0.0]
r22 = Point_F(2)
r22.x = [8.2, 17.6]
r22.face1 = [0.0, 0.0]
r23 = Point_F(2)
r23.x = [5.6, 14.5]
r23.face1 = [0.0, 0.0]
r24 = Point_F(2)
r24.x = [4.2, 11.0]
r24.face1 = [0.0, 0.0]
r25 = Point_F(2)
r25.x = [3.6, 7.0]
r25.face1 = [0.0, 0.0]
r26 = Point_F(2)
r26.x = [4.6, 3.0]
r26.face1 = [0.0, 0.0]
r27 = Point_F(2)
r27.x = [7.0, 1.5]
r27.face1 = [0.0, 0.0]
r28 = Point_F(2)
r28.x = [10.0, 3.2]
r28.face1 = [0.0, 0.0]
r29 = Point_F(2)
r29.x = [12.5, 7.2]
r29.face1 = [0.0, 0.0]
r30 = Point_F(2)
r30.x = [13.5, 11.0]
r30.face1 = [0.0, 0.0]
r31 = Point_F(2)
r31.x = [13.7, 14.8]
r31.face1 = [0.0, 0.0]
radius = 3.0
r1.face1_radius = radius
r2.face1_radius = radius
r3.face1_radius = radius
r4.face1_radius = radius
r5.face1_radius = radius
r6.face1_radius = radius
r7.face1_radius = radius
r8.face1_radius = radius
r9.face1_radius = radius
r10.face1_radius = radius
r11.face1_radius = radius
r12.face1_radius = radius
r13.face1_radius = radius
r14.face1_radius = radius
r15.face1_radius = radius
r16.face1_radius = radius
refpoints = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31]
refCloud = PointCloud( 2, xs=[], points=refpoints)
refs = refCloud.xs
# パターンマッチング試験データ終わり
################################


def plotdata(ax,closest):
    rot = closest.rotation_best[0]
    sin = math.sin(rot)
    cos = math.cos(rot)
    sift = closest.sift_best
    magnify = closest.magnify_best
    centre_ref = closest.size_ref["centre"]
    for i in range(len(xsdata)):
        x = (xsdata[i][0] + sift[0])*magnify[0]*cos - (xsdata[i][1] + sift[1])*magnify[1]*sin
        y = (xsdata[i][1] + sift[1])*magnify[1]*cos + (xsdata[i][0] + sift[0])*magnify[0]*sin
        ax.scatter(x,y, c="black", s=10.0, marker='o')
    for i in range(len(refs)):
        xr = refpoints[i].x[0] - centre_ref[0]
        yr = refpoints[i].x[1] - centre_ref[1]
        ax.scatter(xr,yr, c="red", s=20.0, marker='o')

############### 点群マッチングのテストプログラム ################
def main():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.show()
    closest = Closest(2, xsdata, refs)
    mat = 0.0
    fig.show()
    resolution = 1.

    # 乱数による探索
    for i in range(10):
        plt.cla()
        plotdata(ax,closest)
        fig.show()
        plt.pause(0.01) #これが無いと何故か表示されない
        mat = closest.closest_point_search( search_num=10, resolution=resolution, matching_func_best=mat)
        print(mat)
    
    # ここから、グラジェントに基づいて行う
    cloud = PointCloud(2, xs=xsdata)
    cloud.points_line_sort()
    closest.trans_best_to_default()
    dist_mean = cloud.dist_mean
    print('dist_mean')
    print(dist_mean)
    rot_scale = dist_mean/cloud.cloud_radius*0.0001
    x_scale   = dist_mean*0.0001
    mag_scale = dist_mean/cloud.cloud_radius*0.0001
    d_x = [0.0, 0.0]
    d_rot = [0.0, 0.0]
    d_mag = [0.0, 0.0]
    for i in range(100):
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
            d_x[n]   = x_scale*diff_x[n]/diff_x_scale*20.
            d_rot[n] = rot_scale*diff_rot[n]/diff_rot_scale*400.
            d_mag[n] = mag_scale*diff_mag[n]/diff_mag_scale*200.
        mat, is_converge = closest.closest_point_approach( resolution, mat, d_x, d_rot, d_mag)
        closest.trans_best_to_default()
        print(mat, is_converge)
        if is_converge:
            break
    plt.cla()
    plotdata(ax,closest)
    fig.show()
    plt.pause(0.01) #これが無いと何故か表示されない
    print('Ended')
    plt.show()



def plotdata2(ax, xs):
    i = 0
    for xp in xs:
        x = xp[0]
        y = xp[1]
        ax.scatter(x,y, c="black", s=10.0, marker='o')
        ax.text(x, y, str(i))
        i += 1

################ 点群ソート、分割のテストプログラム ########################3
def main2():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.show()
    plt.cla()
    cloud = PointCloud(2, points=refpoints) # これで正しいはず
    plotdata2(ax,cloud.xs)
    fig.show()
    plt.pause(1.0) #これが無いと何故か表示されない
    #print(cloud.far_most_num)
    
    #print(cloud.points)
    cloud.points_line_sort()
    #cloud.points_line_sort() # ２回実施することで、ベクトル補正に基づいた再ソートが行われる・・・現在不調
    #print("Sorted data\n")
    #print(cloud.points)
    plt.cla()
    plotdata2(ax,cloud.xs)
    fig.show()
    plt.pause(1.0) #これが無いと何故か表示されない
    
    new_clouds = cloud.cloud_division()
    for new_cloud in new_clouds:
        plt.cla()
        plotdata2(ax,new_cloud.xs)
        fig.show()
        plt.pause(3.0) #これが無いと何故か表示されない
    plt.show()



if __name__ == '__main__':
    main()