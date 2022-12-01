# 点群データの扱い 
# AIノートの21/March/2021以降の考察に基づく
# 新たに、座標変換による類似度評価を新たなクラスとするもの。
#
# 2022Apr15 点群マッチングの全面numpy化 Ver.7
# 2022Apr15 点群マッチングの評価体系を変え、参照点群refの一部が重複してデータ点群xsの点に一致しても点数が増えない仕様にした
#  これによって、データ点群をぺしゃんこに潰したものに収束することが無くなった
#  また、任意の次元に対して適用可能で、変形の自由度も増えた
# 2022Feb10 刷新してVer.6
# 2022Mar21 時系列点群の扱いを追加
# 2022Mar21 Closest の演算にnumpyを導入

# 一連のライブラリを使った画像の一致検討は、karaageディレクトリにて、pict2points2.py と pict2points5.py で行っている
# このバージョンによるトライは、pict2points5.py

#from matplotlib import pylab as plt  # 非推奨
import matplotlib.pyplot as plt
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
        centre[n] = x_sum[n]/max(1, len(xs))  # 点群の中心位置
    return centre

# 平行移動した点群のxsリストのみ返す 2022Feb7追加
def shift_cloud(dimension, xs, shift):
    new_xs = []
    for x_org in xs:
        new_x = create_dimensional_list(dimension, 0.0)
        for n in range(dimension):
            new_x[n] = x_org[n] + shift[n]
        new_xs.append(new_x)
    return new_xs

# point_F クラスを定義
# 単独の点のデータフォーマット
# 30/April/2021 の定義に従う
# 5次元空間まで扱う → 見直して、無制限に
# face は「顔の向いてる方向」のベクトル
# 見直し → # ポイントクラウド（点群）は、単に点のリスト
# 見直し → # 並べ替えや再グループ化を自由に行うためには、クラウドは単にリストの方が良い
# 見直し 自由に使えるフラグを廃止し、Cloudに移した 2021Oct21
class Point_F:
    def __init__(self, dimension, time_now=False):
        if time_now==False:
            time_now=time.time()
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
        self.time_stamp = time_now   # 新設2022Mar21
        self.attribute = [] #2022Apr8 新設。属性情報等汎用に使える

############################################
# 点群のクラス。名称とコード上の位置変更。
# 仕様変更し、単純に点の座標リストと、Point_Fのリストのどちらも受け入れる。
# つまり、少数点群にも多数点群にも対応できる
# points は、 Point_F のリスト
# xs は、[[x,y,z..],[x,y,z,..],[x,y,z,..],..] の形
# 両方が入力されちゃったら、pointsが優先されるので注意
# ミュータブルな変数はデフォルトにできないので、xs, points の値が無い場合は [] を入力する事が必要なので注意
# 2022Feb10 dist_mean が2乗平均だったのを変更し、sqrtするように変更。
class PointCloud:
    def renew_cloud(self): # 新設 2022Feb10
        self.len = len(self.xs)   # 点の数
        self.centre = create_dimensional_list(self.dim, 0.0)
        self.vector_cloud = []  # points_line_sort() を実施して書き込まれる。点間相対位置ベクトル。点群より一つ数が減るので別のリストとする。別ルーチンで書き込まれる
        self.distance = []      # 点間相対距離。点群より一つ数が減るので別のリストとする。別ルーチンで上記vector_cloudと共に書き込まれる
        self.dist_mean = 0.0    # 上記点間相対距離の平均値。一筆書きにソートしないと評価できないので、上記別ルーチンで書き込まれる
        # 点群の中心位置を評価する
        self.centre = cloud_centre(self.xs, self.dim)
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
            for n in range(self.dim):
                radius += (self.xs[num][n]-self.centre[n])**2
            radius_total += radius
            if radius > max_radius:
                far_most_num = num
                max_radius = radius
        self.far_most_num = far_most_num    # 中心から最も遠い点の番号
        self.cloud_radius = radius_total/max(1,self.len)   # 点群の大雑把な半径

    def __init__(self, dimension, xs, points):
        self.ID = 0 # デフォルトのゼロは「ＩＤ無し」を意味する。
        self.points = points  # 点群本体（Point_Fのリスト）
        if points==[]:
            self.xs = xs
        else:
            self.xs = []
            for point in points:
                self.xs.append(point.x) # xsは必ず入ってる
        self.dim = dimension    # 次元
        # ここから仮入力
        self.len = 0 #len(self.xs)   # 点の数
        #self.centre = create_dimensional_list(dimension, 0.0)
        self.vector_cloud = []  # points_line_sort() を実施して書き込まれる。点間相対位置ベクトル。点群より一つ数が減るので別のリストとする。別ルーチンで書き込まれる
        self.distance = []      # 点間相対距離。点群より一つ数が減るので別のリストとする。別ルーチンで上記vector_cloudと共に書き込まれる
        self.dist_mean = 0.0    # 上記点間相対距離の平均値。一筆書きにソートしないと評価できないので、上記別ルーチンpoints_line_sortで書き込まれる
        self.centre = []#cloud_centre(self.xs, dimension)
        #max_radius = 0.0
        #radius_total = 0.0
        #far_most_num = 0    # 点の数が１個だった場合は必要
        #for num in range(self.len):
        #    radius = 0.0
        #    for n in range(dimension):
        #        radius += (self.xs[num][n]-self.centre[n])**2
        #    radius_total += radius
        #    if radius > max_radius:
        #        far_most_num = num
        #        max_radius = radius
        self.far_most_num = 0 #far_most_num    # 中心から最も遠い点の番号
        self.cloud_radius = 0.0 #radius_total/max(1,self.len)   # 点群の大雑把な半径
        self.flag1 = create_dimensional_list(self.len, False)  # 自由に使えるフラグ
        self.flag2 = create_dimensional_list(self.len, False)  # 自由に使えるフラグ
        self.times = []     # 新設2022Mar21
        self.renew_cloud()


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
        # 点群が空だった場合に備えて改装 2022Feb7
        new_xs=[]
        new_points=[]
        if len(self.times)==len(self.xs):
            times_list = True
        else:
            times_list = False
        new_times = []  # 2021Mar21 追加
        if self.points != []:
            new_points = [self.points[self.far_most_num]]
        new_xs = [self.xs[self.far_most_num]]   # 並べ直した点群の１番目は、最も中央から遠い点
        if times_list:
            new_times = [self.times[self.far_most_num]]    # 2021Mar21 追加
        # 点群が空だった場合に備えて改装 end 2022Feb7
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
            if self.points != []:
                new_points.append(self.points[j_near])
            new_xs.append(self.xs[j_near])
            if times_list:
                new_times.append(self.times[j_near])    # 2021Mar21 追加
            self.flag1[i] = 1
            self.flag1[j_near] = 1
            i = j_near
        #new_points.append(self.points[j_near])    # 一番最後の点
        self.points = new_points # ソートした点群に置き換え
        self.xs = new_xs  # ソートした点群に置き換え
        if times_list:
            self.times = new_times  # 2021Mar21 追加
        self.far_most_num = 0   # 最も遠い点は１番目になる
        self.dist_mean = math.sqrt(dist_total/max(1,self.len)) # 点群が空だった場合に備えてmax 2022Feb7 # sqrtに変更 2022Feb10


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
        if len(self.times)==len(self.xs):
            times_list = True
        else:
            times_list = False
        for edge in edges:  # 点群を分割する
            #print('edge', edge)
            #print('prev', prev)
            new_cloud = copy.copy(self) # パラメータ類を継承するために先ず複製
            if self.points != []:
                new_cloud.points = self.points[prev:edge+1]
            else:
                new_cloud.points = []
            new_cloud.xs = self.xs[prev:edge+1]
            if times_list:
                new_cloud.times = self.times[prev:edge+1]
            new_cloud.vector_cloud = self.vector_cloud[prev:edge]
            new_cloud.distance = self.distance[prev:edge]
            new_cloud.len = edge + 1 - prev
            new_cloud.centre_and_far_most() # 中心位置などを再評価 print
            new_clouds.append(new_cloud)    # 分割した点群を点群のリストとして納める
            prev = edge+1
        #print('div end')
        if edges:   # 分割した最後の部分
            new_cloud = copy.copy(self) # パラメータ類を継承するために先ず複製
            if self.points != []:
                new_cloud.points = self.points[prev:]
            else:
                new_cloud.points = []
            new_cloud.xs = self.xs[prev:]
            if times_list:
                new_cloud.times = self.times[prev:]
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
# 完全一致の場合と、点群が1つの点しか無かった場合の対応中 2022Feb7
# さらに、全ての点が同じ位置にある場合の対応が必要
class Closest:

    # 評価関数本体。高速化のために現在不使用なので注意
    # 一致度が高い（point 同士の距離が近い）と、高い値を返す
    # 全面numpy化にともなって引数を変えたので注意
    # 最大値は1
    def matching_function(self, diff2):
        match = self.resolution**2 /(diff2 + self.resolution**2)
        return(match)

    def __init__(self, dimension, xs, ref ):
        #print("before init")
        self.xs = np.array(xs)    # 点座標のリスト
        self.ref = np.array(ref)  # 点座表のリストで、動かない参照点
        self.dimension = dimension
        sift = []
        magnify = []
        if len(xs) > 1 and len(ref) > 1:
            self.size_cloud = cloudSize(xs,  dimension)
            self.size_ref   = cloudSize(ref, dimension)
            #rotation = []
            scale_cloud = 0.0
            scale_ref = 0.0
            for n in range(dimension):
                scale_cloud += self.size_cloud["scale"][n]
                scale_ref += self.size_ref["scale"][n]
            if scale_cloud==0.0:
                scale_fact = 1.
            else:
                scale_fact = scale_ref/scale_cloud
            for n in range(dimension):
                sift.append(- self.size_cloud["centre"][n])
                magnify.append(scale_fact)
                #rotation.append( 0.0 )
            self.scale_fact = scale_fact    # 大雑把なスケール差のパラメータ
            self.sift = np.array(sift)
            self.transform = np.eye(dimension)*np.linalg.norm(magnify)
            #self.magnify = magnify
            #self.rotation = rotation
            
            self.sift_best = copy.copy(self.sift)
            self.transform_best = np.eye(dimension)*np.linalg.norm(magnify)
            #self.magnify_best = copy.copy(self.magnify)
            #self.rotation_best = copy.copy(self.rotation)
        else:
            self.size_cloud = 0. 
            self.size_ref   = 0. 
            #self.magnify = []
            #self.rotation = []
            self.scale_fact = 1.
            for n in range(dimension):
                self.sift.append(- self.size_cloud["centre"][n])
                magnify.append(self.scale_fact)
                #self.rotation.append( 0.0 )
            self.sift = np.array(sift)
            self.transform = np.eye(dimension)*np.linalg.norm(magnify)
            self.sift_best = copy.copy(self.sift)
            self.transform_best = np.eye(dimension)*np.linalg.norm(magnify)
            #self.magnify_best = copy.copy(self.magnify)
            #self.rotation_best = copy.copy(self.rotation)
        # 部分一致をここでやると、おかしなことになるので、MAX_MAGNIFICATION の値は控えめにして、部分一致はグループの再構成で対応する事にする
        self.MAX_MAGNIFICATION = 2.0
        self.RATE_UPDATE_SCALE = 0.8
        self.resolution = 1.0       # インスタンスに入れた 2022Apr15
        self.u_mutching_f = np.vectorize(self.matching_function)
        #self.u_mutching_f = np.vectorize(matching_function_)
        self.R_xs = np.ones(len(self.ref))
        self.R_ref = np.ones(len(self.xs))
        self.ref_matrix = np.tensordot(self.R_ref, self.ref, axes=0).transpose(1,0,2)
        #print("after ref_matrix")

    # numpy使った変換 2022Mar21 次元数は自由
    # ごっそりnumpy化 2022Apr15 Ver.7
    def coordinate_transformation(self):
        sift_R = np.tensordot(self.R_ref, self.sift, axes=0)
        transed_points = np.dot(self.xs + sift_R, self.transform)
        return transed_points

    # 2022Mar21新設 refを逆変換
    # 2022Apr15 全面numpy化
    def coordinate_antitransformation(self, ref_org):
        centre_ref = np.tensordot(np.ones(len(ref_org)), self.size_ref["centre"], axes=0)
        transed_refs = np.dot(np.linalg.inv(self.transform), ref_org-centre_ref)
        return transed_refs

    # 特定の座標変換に対して、一致度を評価する
    # refferenceは変換しない
    # resolution のスケールは、拡大率の異方性を考慮せず、scale_factのみで補正する
    # 参照点群ref の各点の、face方向にデータ点群の点が入っていたら距離を減らす補正をして、参照点のつながりを表現する。
    #  この処理は2次元にしか対応しておらず、再考を要する。
    # 廃止 -> 完全一致の場合に PERFECT_MATCHING を返す仕様追加 2022Feb7
    # PERFECT_MATCHING = 2.0
    def closest_point_evaluation(self):
        transed_points = self.coordinate_transformation()
        centre_xs = np.tensordot(np.ones(len(self.xs)), self.size_ref["centre"], axes=0)
        xs_sift = transed_points + centre_xs
        xs_matrix  = np.tensordot(self.R_xs,  xs_sift,  axes=0)
        diff_matrix = xs_matrix - self.ref_matrix
        diff_matrix_ab = np.sum(np.power(diff_matrix, 2), axis=2)

        # 評価関数
        # 高速化のために、ユニバーサル化関数を使わない（ユニバーサル化関数の計算速度は for 文と同じ）
        matching = self.resolution**2 /(diff_matrix_ab + self.resolution**2)
        #matching = self.u_mutching_f(diff_matrix)
        #print(np.shape(matching))
        #print(len(self.xs))

        matching_func_ref_max = np.max(matching, axis=1)    # xsの各点の最大の評価関数を抜き出す
        #matching_func_total = np.sum(matching)             # Ver.6 相当のアルゴリズム
        matching_func_total = np.sum(matching_func_ref_max)
        # numpy不使用(～Ver.6)のアルゴリズム
        #matching_func_total = 0.0
        #for new_x in transed_points:
        #    matching_func_point = 0.0
        #    for x_r in self.ref:
        #        diff_vector = create_dimensional_list(self.dimension, 0.0)
        #        diff2 = 0.0
        #        face_diff = 0.0
        #        for n in range(self.dimension):
        #            diff_vector[n] = new_x[n] - x_r[n] + self.size_ref["centre"][n]
        #            diff2 += diff_vector[n]**2
        #            # face方向に入っていたら距離を減らす補正をして、点のつながりを表現する・・・廃止
        #            #face_diff += (diff_vector[n])*(point_r.face1[n])
        #        if diff2==0.0:
        #            #print('xs length = ' + str(len(self.xs)) + ',   ref length = ' + str(len(self.ref)) )
        #            return self.PERFECT_MATCHING
        #        diff2 /= max(1.0,face_diff/math.sqrt(diff2)) 
        #        matching_func_point += self.matching_function(diff2, resolution) # *self.scale_fact) # 2022Feb4 修正
        #    matching_func_total += matching_func_point

        #matching_func_norm = matching_func_total/max(len(self.xs), len(self.ref) ) # Ver.6 相当のアルゴリズム
        matching_func_norm = matching_func_total/len(self.ref)
        return(matching_func_norm)


    # 座標変換のパラメータをランダムに作成
    # rate_update = 1.0 で完全ランダム
    # rate_update = 0.0 で、変換を固定
    # メモ、random.random() は 0.0 から 1.0 の乱数 
    # 回転と拡大縮小にしているので、この関数は2次元でしかまともに動かない
    #  ・・・SLAM向けとして、今後n次元版を開発
    def transformation_random(self, rate_update, magnify=True):
        rot_new =  (2*random.random()-1.0)*math.pi
        c_new = math.cos(rot_new)
        s_new = math.sqrt(1. - c_new**2)*np.sign(random.random()-0.5)
        if magnify:
            mag_new = self.scale_fact*(self.MAX_MAGNIFICATION**(2*random.random()-1.0)) # メモ 「**」はべき乗
        else:
            mag_new = 1.
        for n in range(self.dimension):
            sift_new = - self.size_cloud["centre"][n] + (random.random()-0.5)*self.scale_fact
            self.sift[n] = sift_new*rate_update + self.sift_best[n]*(1.0-rate_update)
            for m in range(self.dimension):    
                # MAX_MAGNIFICATION倍～1/MAX_MAGNIFICATION の範囲で探索する
                #mag_new = self.scale_fact*(self.MAX_MAGNIFICATION**(2*random.random()-1.0)) # メモ 「**」はべき乗
                if n==m:
                    trans_new = c_new*mag_new
                elif n>m:
                    trans_new = s_new*mag_new
                else:
                    trans_new = -s_new*mag_new
                self.transform[n,m] = trans_new*rate_update + self.transform_best[n,m]*(1.0-rate_update)


    # ぺしゃんこに潰してしまう変換に落ちてしまう事を防ぐことに成功したので、
    # 変換行列の要素を全て乱数で作る仕様を作成
    # 任意の次元数で使える
    def transformation_random_ful(self, rate_update):
        for n in range(self.dimension):
            sift_new = - self.size_cloud["centre"][n] + (random.random()-0.5)*self.scale_fact
            self.sift[n] = sift_new*rate_update + self.sift_best[n]*(1.0-rate_update)
            for m in range(self.dimension):    
                trans_new = self.scale_fact*(self.MAX_MAGNIFICATION*(2*random.random()-1.0))
                self.transform[n,m] = trans_new*rate_update + self.transform_best[n,m]*(1.0-rate_update)

    # 最善の変換条件を記録
    def trans_best(self):
        self.sift_best = copy.deepcopy(self.sift)
        self.transform_best = copy.deepcopy(self.transform)
    
    # 最善の変換条件をダウンロード
    def trans_best_to_default(self):
        self.sift = copy.deepcopy(self.sift_best)
        self.transform = copy.deepcopy(self.transform_best)

    # 様々な座標変換を試し、一致度探索を一定回数行う
    # resolution のアップデートはここでは行わず、別途行う（問題の性質によって様々な応答がありうるため
    # transmode='full' は、変形を含む
    # transmode='rot' は、回転のみ（SLAM等）
    # transmode='mag' は、変形の無い拡大縮小と回転
    def closest_point_search(self, search_num, matching_func_best, transmode='full'):
        # rate_update = 1.0 で完全ランダム
        # rate_update = 0.0 で、変換を固定
        # rate_update の決め方を、point_cloud2.pyから変えた
        rate_update = 1.0 
        for i in range(search_num):
            if transmode=='full':
                self.transformation_random_ful(rate_update)
            elif transmode=='rot':
                self.transformation_random(rate_update, magnify=False)
            else:
                self.transformation_random(rate_update, magnify=True)
            matching_func_evaluation = self.closest_point_evaluation()
            #print(matching_func_evaluation)
            if matching_func_evaluation > matching_func_best:
                # 良い一致が得られたら、座標変換パラメータの変化を少なくする
                rate_update *= max(self.RATE_UPDATE_SCALE, matching_func_best/matching_func_evaluation)
                matching_func_best = matching_func_evaluation
                self.trans_best()
                #print(self.transform_best)
        return(matching_func_best)

    # 各変換に対する微分係数を求める 2021Oct21 追加
    # ピーク位置を推定するアルゴリズム追加 2022Mar21
    def closest_point_diff(self, x_scale, mag_scale):
        sift_prev = copy.copy(self.sift)
        mag_prev = copy.copy(self.transform)
        #diff_rot = []
        diff_x = []
        diff_mag = np.eye(self.dimension)
        #diff_rot_m = []
        diff_x_m = []
        diff_mag_m = np.eye(self.dimension)
        matching_func_norm = self.closest_point_evaluation()
        nonzero_sift = False
        nonzero_mag = False
        for n in range(self.dimension):
            self.sift[n] += x_scale
            matching_func = self.closest_point_evaluation()
            self.sift[n] += x_scale
            matching_func2 = self.closest_point_evaluation()
            if matching_func2==matching_func: # ゼロ割防止
                diff_x.append(0.)
                diff_x_m.append(0.)
            else:
                diff_x.append((matching_func - matching_func_norm)/x_scale)
                c = (matching_func_norm-matching_func2)*mag_scale/(matching_func2-matching_func*2.+matching_func_norm)/2.
                diff_x_m.append(x_scale + c)
                nonzero_sift = True
            self.sift[n] = sift_prev[n]

            for m in range(self.dimension):
                self.transform[n,m] += mag_scale
                matching_func = self.closest_point_evaluation()
                self.transform[n,m] += mag_scale
                matching_func2 = self.closest_point_evaluation()
                if matching_func2==matching_func: # ゼロ割防止
                    diff_mag[n,m] = 0.
                    diff_mag_m[n,m] = 0.
                else:
                    diff_mag[n,m] = (matching_func - matching_func_norm)/mag_scale
                    c = (matching_func_norm-matching_func2)*mag_scale/(matching_func2-2.*matching_func+matching_func_norm)/2.
                    diff_mag_m[n,m] = mag_scale + c
                    nonzero_mag = True
                self.transform = copy.copy(mag_prev)
        # 新アルゴリズム ピーク位置推定
        # diff_x_m, diff_mag_m は共にピーク位置の推定値だが、おかしな値の可能性がある
        # そこで、先ず大きさを diff_x_scale, diff_mag_scale で規制する
        diff_x_scale   = min(x_scale,   np.linalg.norm(np.array(diff_x_m)))
        diff_mag_scale = min(mag_scale, np.linalg.norm(diff_mag_m))
        # 次いで、微分値の成分である diff_x, diff_mag に平行で、大きさが diff_x_scale, diff_mag_scale になるように置き換える
        if nonzero_sift:
            diff_x   = np.array(diff_x)
            diff_x   = diff_x/np.linalg.norm(diff_x)*diff_x_scale
        if nonzero_mag:
            diff_mag = np.array(diff_mag)
            diff_mag = diff_mag/np.linalg.norm(diff_mag)*diff_mag_scale
        return diff_x, diff_mag


    def closest_point_approach(self, matching_func_best, diff_x, diff_mag, tolerance=0.0):
        best = matching_func_best*(1.0 + tolerance)
        x_scale   = np.linalg.norm(diff_x)
        mag_scale = np.linalg.norm(diff_mag)

        sift_pre = copy.copy(self.sift)
        self.sift = self.sift + diff_x
        matching_func_sift = self.closest_point_evaluation()
        if x_scale>0. and matching_func_sift > best: 
            is_converge_sift = False
        else:
            is_converge_sift = True
            x_scale = x_scale*0.
        self.sift = sift_pre

        trans_pre = copy.copy(self.transform)
        self.transform = self.transform + diff_mag
        matching_func_mag = self.closest_point_evaluation()
        if mag_scale>0. and matching_func_mag > best: 
            is_converge_mag = False
        else:
            is_converge_mag = True
            mag_scale = mag_scale*0.
            self.transform = trans_pre

        if is_converge_sift:
            if is_converge_mag:
                matching_func_evaluation = matching_func_best
            else: #is_converge_mag==False and is_converge_diff==True
                matching_func_evaluation = matching_func_mag
        else:
            self.sift = self.sift + diff_x
            if is_converge_mag:#is_converge_mag==True and is_converge_diff==False
                matching_func_evaluation = matching_func_sift
            else: #is_converge_mag==False and is_converge_diff==False
                matching_func_evaluation = self.closest_point_evaluation()
        self.trans_best()
        return matching_func_evaluation, is_converge_sift, is_converge_mag, x_scale, mag_scale


    # 包括的なグラジェントアプロ－チのセット
    def closest_gradient_approach(self, x_scale, mag_scale, number, tolerance=0.0):
        mat = self.closest_point_evaluation()
        for converge_num in range(number):
            diff_x, diff_mag = self.closest_point_diff( x_scale, mag_scale)
            mat, is_converge_sift, is_converge_mag, x_scale, mag_scale = self.closest_point_approach( mat, diff_x, diff_mag, tolerance)
            self.trans_best_to_default()
            if is_converge_sift and is_converge_mag:
                break
        return converge_num, mat, is_converge_sift, is_converge_mag #, x_scale, mag_scale
        

        


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
#xsdata = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]

pl1  = [150., 130.]
pl2  = [130., 70.]
pl3  = [110., 30.]
pl4  = [ 60., 10.]
pl5  = [ 25., 60.]
pl6  = [ 50., 110.]
pl7  = [100., 160.]
pl8  = [180., 170.]
pl9  = [230., 120.]
pl10 = [220., 40.]
pl11 = [230., 80.]
pl12 = [210., 150.]
pl13 = [140., 170.]
pl14 = [ 75., 140.]
pl15 = [ 35., 90.]
pl16 = [ 23., 35.]
pl17 = [ 33., 12.]
pl18 = [ 85., 17.]
pl19 = [120., 53.]
pl20 = [143., 100.]
xsdata = [pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8,pl9,pl10,pl11,pl12,pl13,pl14,pl15,pl16,pl17,pl18,pl19,pl20]

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
    #rot = closest.rotation_best[0]
    #sin = math.sin(rot)
    #cos = math.cos(rot)
    sift = closest.sift_best
    centre_ref = closest.size_ref["centre"]
    xsx = []
    xsy = []
    for i in range(len(xsdata)):
        xsr = np.dot(np.array(xsdata[i]) + sift, closest.transform_best)
        #xr = np.dot(closest.transform, np.array(xsdata[i]) + sift)
        #x = (xsdata[i][0] + sift[0])*magnify[0]*cos - (xsdata[i][1] + sift[1])*magnify[1]*sin
        #y = (xsdata[i][1] + sift[1])*magnify[1]*cos + (xsdata[i][0] + sift[0])*magnify[0]*sin
        xsx.append(xsr[0])
        xsy.append(xsr[1])
    ax.scatter(xsx,xsy, c="black", s=10.0, marker='o')
    xrx = []
    xry = []
    for i in range(len(refs)):
        xr = refpoints[i].x[0] - centre_ref[0]
        yr = refpoints[i].x[1] - centre_ref[1]
        xrx.append(xr)
        xry.append(yr)
    ax.scatter(xrx,xry, c="red", s=20.0, marker='o')

def plotdata_anti(ax,closest):
    #magnify = closest.magnify_best
    centre_ref = np.array(closest.size_ref["centre"])
    #sin = math.sin(closest.rotation_best[0])
    #cos = math.cos(closest.rotation_best[0])
    #siftnp = np.array(closest.sift_best)
    #trans = np.array([[magnify[0]*cos, -magnify[1]*sin], \
    #                  [magnify[1]*sin,  magnify[0]*cos]])
    transinv = np.linalg.inv(closest.transform_best)
    xsx = []
    xsy = []
    for xs in xsdata:
        xsx.append(xs[0])
        xsy.append(xs[1])
    ax.scatter(xsx,xsy, c="black", s=10.0, marker='o')
    xrx = []
    xry = []
    for ref in refs:
        refnp = np.array(ref)
        xr = np.dot(refnp - centre_ref,transinv) - closest.sift_best
        xrx.append(xr[0])
        xry.append(xr[1])
    ax.scatter(xrx,xry, c="red", s=20.0, marker='o')

############### 点群マッチングのテストプログラム ################
def main():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.show()
    closest = Closest(2, xsdata, refs)
    mat = 0.0
    fig.show()
    # resolution の値は、1.0 で検討しているが、大きくするとローカルに落ちる事が無くなった。
    # ただしその場合は、合わない点に引っ張られることになる。
    closest.resolution = 1.

    # 乱数による探索
    plt.cla()
    #plotdata(ax,closest)
    plotdata_anti(ax,closest) # refの方を座標変換
    mat_prev = 0.
    for i in range(100):
        mat = closest.closest_point_search( search_num=100, matching_func_best=mat, transmode='full')
        if mat > mat_prev:
            plt.cla()
            #plotdata(ax,closest)
            plotdata_anti(ax,closest) # refの方を座標変換
            fig.show()
            plt.pause(0.001) #これが無いと何故か表示されない
            print('at i=%d, matching = %f'%(i, mat))
        mat_prev = mat

    # ここから、グラジェントに基づいて探索
    # 点間隔をcloud.dist_mean として評価するためにPointCloudを使う。点間隔がgivenならこの演算は不要
    cloud = PointCloud(2, xs=xsdata, points=[])
    cloud.points_line_sort()
    closest.trans_best_to_default()
    print('dist_mean = %.4f' % cloud.dist_mean)
    # dist_meanの意味をVer5から変えたので、まともな演算になった・・・点間隔の1/10ぐらいにscaleを設定
    #rot_scale = cloud.dist_mean/cloud.cloud_radius*0.1
    x_scale   = cloud.dist_mean*1 
    mag_scale = cloud.dist_mean/cloud.cloud_radius*1 
    #print(rot_scale)
    
    # 下記の for 文とclosest_gradient_approach関数は同じ
    #for i in range(300):
    #    #plt.cla()
    #    ##plotdata(ax,closest)
    #    #plotdata_anti(ax,closest) # refの方を座標変換
    #    #fig.show()
    #    #plt.pause(0.001) #これが無いと何故か表示されない
    #    # closest_point_diff関数で precision=False にするか、closest_point_approach(関数で tolerance=0.001 にしないと収束判定できない
    #    diff_x, diff_mag = closest.closest_point_diff( x_scale, mag_scale, precision=True)
    #    mat, is_converge, x_scale, mag_scale = closest.closest_point_approach( mat, diff_x, diff_mag, tolerance=0.00) #tolerance=0.001)
    #    closest.trans_best_to_default()
    #    #print('diff', diff_x, diff_mag)
    #    #print(mat, is_converge)
    #    if is_converge:
    #        break
    converge_num, mat, is_converge_sift, is_converge_mag = closest.closest_gradient_approach(x_scale, mag_scale, number=300, tolerance=0.0)
    plt.cla()
    #plotdata(ax,closest)
    plotdata_anti(ax,closest) # refの方を座標変換
    fig.show()
    plt.pause(0.01) #これが無いと何故か表示されない
    print('n=%d, mat=%f, converge='%(converge_num, mat), is_converge_sift, is_converge_mag)
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
    cloud = PointCloud(2, xs=[], points=refpoints) # これで正しいはず
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