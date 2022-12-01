# IDネットワーク 2021Sep15 作成
import copy
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
# https://note.nkmk.me/python-scipy-sparse-matrix-csr-csc-coo-lil/
# scipy は、デフォルトでは入ってない。仮想環境のターミナル上で、
# pip install --upgrade scipy
# を実行する。karaage には、1.7.1 が入った。

def matrix_power(matrix, power):
    matrix_n = matrix
    for i in range(power-1):
        matrix_n = np.dot(matrix_n, matrix)
    return matrix_n         

############################################################
# IDネットワークのクラス本体記述
class ID_NetWork:
    # モノを検知、生成、または関係性を検出した時の記録を収めるpropertiesの0番データの値と、説明文
    # リストのゼロ番はゼロ番データのダミー値で、1番は説明文
    INITIAL_PROPERY = { \
            "data_type":  ["None", "Data type. The list of types can be obtained from data_type_str()."], \
            "time_stamp": [0.0,    "Time stamp when the participant or relation was created or detected."], \
            "project_ID": [0,      "Project ID that created the participant or detected the relationship."], \
            "app_ID":     [0,      "ID of App that that created the participant or detected the relationship."], \
            "work_ID":    [0,      "ID of work that that created the participant or detected the relationship."]}

    ############# propertiesyのディクショナリを、値もしくは説明文で再構築する #######
    def init_properties(self, num, dictio):
        properties = {}
        for keys_str in dictio.keys():
            properties.update({keys_str: dictio[keys_str][num]}) 
        return(properties)

    # データタイプの一覧。 キーがそのままキーで、値がその説明。
    DATA_TYPE_STRS = {"None": "Nothing. Only for zero's ID is None", \
                "project": "Project", \
                "app": "App", \
                "work": "Work", \
                "unit_data": "Data unit from the memory space", \
                "field": "Field, in other word, stage or place where the data exist", \
                "territory": "Territory, a specific region that is defined by a point cloud.", \
                "graphic": "Graphic pixel data", \
                "movie": "Movie. Usualy, time series of graphic pixel data", \
                "point": "Single point. Usualy obtained by object_detection", \
                "point_cloud_small_num": "Small number point cloud. All of points have to be indexed", \
                "point_cloud_large_num": "Large number point cloud. Indexings are not necessary", \
                "time_series_point_cloud_small_num": "Time series small number point cloud", \
                "time_series_point_cloud_large_num": "Time series large number point cloud", \
                "event": "Event. usualy detected from time seriese data.", \
                "story": "Time series of events." }

    RELATION_TYPE_STRS = {"equal":  ["undirected", "Detected or judged they are same participant"], \
                "matched":   ["directed"  , "Matched to a known thing. Similarlity is also this term. From is the participant, and To is the knowledge(a stereotype or prototype)."], \
                "part":      ["directed"  , "A part of another participant. From is the participant, and To is upper."], \
                "union":     ["undirected", "Two participants were united and became a new participant. This relationship describe relation between the two participants and is undirected."], \
                "merged":    ["directed"  , "Two participants were marged and became a new participant. This relationship describe relation between the two participants and is directed."], \
                "contacted": ["undirected", "Two participants had a physical relationship. Unlike 'predicate', the relation is undirected."], \
                "predicate": ["directed"  , "A predicate of a verb. In other word, target. From is the subject, and To is the predicate. Thus, it's directed unlike 'contacted'."]  ,\
                "not_specified": ["undirected", "Not specified. This is usualy a case that they have relation but the type of relation was not obtained from the relation database."], \
                "non": ["undirected", "No relation was detected. Normally, not resistered in the relation database."]}

    # 対称な関係のリスト
    def sym_relations(self):
        sym_relation_strs = []
        for key in self.RELATION_TYPE_STRS.keys():
            if self.RELATION_TYPE_STRS[key][0]=="undirected":
                sym_relation_strs.append(key)
        return sym_relation_strs

    #SYM_RELATIONS = ["equal", "union", "contacted", "not_specified", "non"]

    ############## __init__ ###############
    def __init__(self):
        ### 先ず、個別の参加者(participant)### 
        self.IDs = [0]                  # ゼロ番目は空席なので埋める
        self.properties = [self.init_properties(0, self.INITIAL_PROPERY)]    # ゼロ番目は空席なので埋める
        ### ここから関係性。関係性そのものにはIDはつかない ###
        self.fromIDs = [0]              # 関係行列の row に相当
        self.toIDs = [0]                # 関係行列の column に相当
        self.relation_strengths = [0.0] # 関係の強さで、値は 1.0 以下で正の実数
        self.relation_types = [[]]      # 中身は文字列。リストの重複に備えてリストのリスト
        self.relation_properties = [[]] # 関係性の特性、データタイプはpropertiesと同じ。リストの重複に備えてリストのリスト 
        self.sym_relation_strs = self.sym_relations() # 対称な関係のリスト

    # 例外処理用。現状ではプリント分のみ
    def error_in_ID(self, str):
        print(str)

    ################################################################
    # 開発者用に、IDsに書き込むべき要素のリストを羅列する
    # キーが properties の要素にあるキーで、値がその説明文。
    def IDs_key_str(self):
        return(self.init_properties(1, self.INITIAL_PROPERY)) # num=1 は説明文

    ################################################################
    # 開発者用に、データタイプを表す文字列の辞書をここに置く。どんどん追加する。
    # キーがそのままデータタイプを表す文字列で、値がその説明文。
    def data_type_str(self):
        return self.DATA_TYPE_STRS 

    #############################################################
    # 開発者用に、関係性を記述する文字列を取得する
    # キーが関係性を示す文字列で、値がその説明文になっている
    def relation_type_str(self):
        return self.RELATION_TYPE_STRS 

    #############################################################
    # ID から、それが何者か（Property）を返す
    def get_property(self, ID):
        return(self.properties[ID])

    ###### 逆の関係を示すために、"inverse_" を付けたり取ったりする #########
    def _inverse_relation(self, relation):
        for sym_str in self.sym_relation_strs: # 対称な関係の場合はそのまま返す
            if relation == sym_str:
                return(relation)
        if relation.find("inverse_") == -1:
            return("inverse_" + relation) # "inverse_" を含まない場合は付加
        else:
            return(relation.strip("inverse_")) # "inverse_" を含む場合は除去

    ##############################################################
    # 関係の有無を返す関数
    # ID1 (FromID) と ID2 (ToID) の関係を返してくれる
    # power は関係行列のpower乗を返すモノで、2以上を指定することで間接的な関係も示してくれるが、関係性は"not_specified"になってしまう
    def is_relation(self, ID1, ID2, power):
        if power==1 :
            list_num = 0
            from_num = 0 
            to_num = 0
            for fromID in self.fromIDs:
                if fromID == ID1 and self.toIDs[list_num] == ID2:
                    to_num = copy.copy(list_num)
                if fromID == ID2 and self.toIDs[list_num] == ID1:
                    from_num = copy.copy(list_num)
                list_num += 1
            if from_num and to_num:
                return({"strength": self.relation_strengths[from_num], \
                        "relation": self.relation_types[from_num], \
                        "from_num": from_num, "to_num": to_num})
            elif from_num: # from だけというのはおかしい
                self.error_in_ID("in is_relation. only from")
            elif to_num: # to だけというのはおかしい
                self.error_in_ID("in is_relation. only to")
            else:
                return({"strength": 0.0, \
                        "relation": "non"})
        else:
            maxID = self.IDs[-1]
            relation_matrix = csr_matrix((self.relation_strengths, (self.fromIDs, self.toIDs)), shape=(maxID+1, maxID+1))
            relation_matrix_n = matrix_power(relation_matrix, power)
            strength = relation_matrix_n[ID1, ID2]
            return({"strength": strength, \
                    "relation": "not_specified"})
    
    ##############################################################
    # 関係性を登録する
    # fromID, toID はID（整数）
    # strength は実数
    # relation は relation_type_str() で一覧が見れる文字列
    # 新しい関係を登録する場合と、既に関係が登録されているところに上書きする場合とで、パターンが違う
    # relation_types と relation_properties はリストのリストになっていて、複数の関係性が登録できるようになってる
    # strength は行列演算を行う為に、リストのリストではなく、大きい方の値で上書き。
    def register_relation(self, fromID, toID, strength, relation, relation_property_dat): # relation_property_dat は INITIAL_PROPERY にある規定のディクショナリであること
        existing_relation = self.is_relation(fromID, toID, 1)
        if existing_relation["relation"] == "non" :
            # 新しい関係だった場合
            self.fromIDs.append(fromID)
            self.toIDs.append(toID)
            self.relation_strengths.append(strength)
            self.relation_types.append([relation]) # リストにリストを追加する
            self.relation_properties.append([relation_property_dat]) # リストにリストを追加する
            # 対称行列にする操作。ただし、fromID と toID が同じの場合（対角成分）は不要
            if fromID != toID :
                self.fromIDs.append(toID)
                self.toIDs.append(fromID)
                self.relation_strengths.append(strength)
                self.relation_types.append([self._inverse_relation(relation)])
                self.relation_properties.append([relation_property_dat])
            return("new_relation_registerd")
        else:
            # 既に関係性が登録されていた場合、strength を上書きし、関係性のリストに追加する
            from_num = existing_relation["from_num"]
            to_num   = existing_relation["to_num"]
            if fromID == self.fromIDs[from_num] and toID == self.toIDs[from_num]:
                self.relation_strengths[from_num] = max(self.relation_strengths[from_num],strength)
                self.relation_types[from_num].append(relation)
                self.relation_properties[from_num].append(relation_property_dat)
                # 対称成分
                self.relation_strengths[to_num] = max(self.relation_strengths[to_num],strength)
                self.relation_types[to_num].append(self._inverse_relation(relation))
                self.relation_properties[to_num].append(relation_property_dat)
                return("relation_appended")
            elif fromID == self.fromIDs[to_num] and toID == self.toIDs[to_num] :
                self.relation_strengths[to_num] = max(self.relation_strengths[to_num],strength)
                self.relation_types[to_num].append(self._inverse_relation(relation))
                self.relation_properties[to_num].append(relation_property_dat)
                # 対称成分
                self.relation_strengths[from_num] = max(self.relation_strengths[from_num],strength)
                self.relation_types[from_num].append(relation)
                self.relation_properties[from_num].append(relation_property_dat)
                return("relation_appended")
            else:
                self.error_in_ID("in register_relation. From and to did not match")
                return("error")

    #############################################################
    # ID を取得し、登録する
    def register_ID(self, property_dat):  # property_dat は INITIAL_PROPERY にある規定のディクショナリであること
        newID = self.IDs[-1] + 1
        self.IDs.append(newID)
        property_dat.setdefault("time_stamp", time.time())
        if property_dat.get("data_type") == "project" :
            property_dat.setdefault("project_ID", newID)
        if property_dat.get("data_type") == "app" :
            property_dat.setdefault("app_ID", newID)
        if property_dat.get("data_type") == "work" :
            property_dat.setdefault("work_ID", newID)
        self.properties.append(copy.copy(property_dat))
        ## 関係行列の対角成分を作る
        ## 対角成分は不要（邪魔）かも知れない。後に検討 2021Sep28 → 廃止決定2021Sep29
        #relation_property =  { \
        #    "data_type":  property_dat.get("data_type",  "None"), \
        #    "time_stamp": property_dat.get("time_stamp", time.time()), \
        #    "project_ID": property_dat.get("project_ID", 0), \
        #    "app_ID":     property_dat.get("app_ID",     0), \
        #    "work_ID":    property_dat.get("work_ID",    0)}
        #self.register_relation(newID, newID, strength=1.0, relation="equal", relation_property_dat=relation_property)
        return(newID)


#####################################################################################################################
##### Test program
if __name__ == "__main__":
    idnet = ID_NetWork()
    project_ID = idnet.register_ID({"data_type": "project"}) # 1
    app_ID =     idnet.register_ID({"data_type": "app"})     # 2
    work_ID =    idnet.register_ID({"data_type": "work"})    # 3

    obj = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID}
    objA=copy.copy(obj)
    objB=copy.copy(obj)
    objC=copy.copy(obj)
    objD=copy.copy(obj)
    objE=copy.copy(obj)
    objF=copy.copy(obj)
    objG=copy.copy(obj)
    objH=copy.copy(obj)
    objA["ID"] = idnet.register_ID(objA) # 4
    objB["ID"] = idnet.register_ID(objB) # 5
    objC["ID"] = idnet.register_ID(objC) # 6
    objD["ID"] = idnet.register_ID(objD) # 7
    objE["ID"] = idnet.register_ID(objE) # 8
    objF["ID"] = idnet.register_ID(objF) # 9
    objG["ID"] = idnet.register_ID(objG) # 10
    objH["ID"] = idnet.register_ID(objH) # 11

    CF =  idnet.register_relation( fromID=objC["ID"], toID=objF["ID"], strength=0.5, relation="part", relation_property_dat=obj)
    print("CF " + CF)
    DG =  idnet.register_relation( fromID=objD["ID"], toID=objG["ID"], strength=0.5, relation="part", relation_property_dat=obj)
    print("DG " + DG)
    CF2 = idnet.register_relation( fromID=objC["ID"], toID=objF["ID"], strength=0.7, relation="part", relation_property_dat=obj)
    print("CF2 " + CF2)
    AG =  idnet.register_relation( fromID=objA["ID"], toID=objG["ID"], strength=0.5, relation="part", relation_property_dat=obj)
    print("AG " + AG)

    relation_AD1 = idnet.is_relation(objD["ID"], objA["ID"], 1)
    print(relation_AD1)
    relation_AD2 = idnet.is_relation(objD["ID"], objA["ID"], 2)
    print(relation_AD2)
