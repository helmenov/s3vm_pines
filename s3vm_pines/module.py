import indianpines as IP
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
import pandas as pd
from importlib import resources
from numpy.random import default_rng

recategorize17to10_csv = resources.files('s3vm_pines')/'recategolize'/'recategorize17to10.csv'

def train_test_split(prop_train = 0.5, recategorize_rule=recategorize17to10_csv, gt_gic=True):
    """
    ラベル付きデータセットをtraining:test = p:1-pに分けて，
    各データ番号に対する status番号 を返す．
    0: background
    1: test
    2: training

    ただし，trainingデータは，隣接pixelで同一カテゴリとなっている領域から選ばれる．

    Output:
    - status (np.array(SampleSize,)): タグ番号リスト
    - status_name (str List(3,)): タグ番号順に並べたタグ名の対応リスト
    """

    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_rule,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : gt_gic,
    }
    pines = IP.load(**IP_conf)
    n_class = pines.target_names.shape[0]

    #lx = np.max(pines.cordinates[:,0])+1
    #ly = np.max(pines.cordinates[:,1])+1

    def xy2idx(x,y):
        ly = np.max(pines.cordinates[:,1])+1
        return y + x*ly

    claster_list = list()
    for t in range(1,n_class):
        claster = np.zeros_like(pines.target)
        LookupTable = list()
        label = 0
        LookupTable.append(label)
        for xi, yi in pines.cordinates[pines.target==t]:
            c_Neighbors = list()
            # UpperLeft
            if xi-1 > 0 and yi-1 >0: c_Neighbors.append(claster[xy2idx(xi-1,yi-1)])
            # Upper
            if xi > 0 and yi-1 >0: c_Neighbors.append(claster[xy2idx(xi,yi-1)])
            # Left
            if xi-1 > 0 and yi > 0: c_Neighbors.append(claster[xy2idx(xi-1,yi)])
            # UnderLeft
            if xi-1 > 0 and yi+1 >0: c_Neighbors.append(claster[xy2idx(xi-1,yi+1)])
            c_Neighbors = np.array(c_Neighbors)
            #print(c_Neighbors)
            if np.all(c_Neighbors == 0):
                label += 1
                LookupTable.append(label)
                claster[xy2idx(xi,yi)] = label
                #print(f"({xi},{yi}) is labeled {label} [A]")
            else:
                if len(c_Neighbors[c_Neighbors>0]) > 0:
                    claster[xy2idx(xi,yi)] = min(c_Neighbors[c_Neighbors>0])
                    #print(f"({xi},{yi}) is labeled {min(c_Neighbors[c_Neighbors>0])}[B]")
                    #print(f"claster[{xy2idx(xi,yi)}] = {claster[xy2idx(xi,yi)]}")
                    for c_other in c_Neighbors[c_Neighbors>claster[xy2idx(xi,yi)]]:
                        LookupTable[c_other] = claster[xy2idx(xi,yi)]
            #print("---")

        # LookupTableを降順に見て，使われなかったラベルを統合
        L = len(LookupTable)
        for i,j in enumerate(LookupTable[::-1]):
            if j != L-i-1:
                claster[claster==L-i-1] = j

        # indexlistを作る
        claster_list_t = list()
        for c in range(1,max(claster)+1):
            list_idx = sorted([idx for idx in range(len(claster)) if claster[idx]==c])
            claster_list_t.append(list_idx)
            #print(len(claster[claster == c]))
        #print(claster_list_t)
        claster_list.append(claster_list_t)
        #print('='*10)

    # claster_list[t] : targetのクラスター番号リスト
    # claster_list[t][l] : targetのl番目のクラスターのインデクスリスト

    status = np.zeros_like(pines.target)
    for t in range(1,n_class):
        idx = (pines.target==t)
        status[idx] = 1
        n_labeled = pines.target[pines.target == t].shape[0]
        #print(n_labeled)
        n_train = int(np.ceil(prop_train * n_labeled))
        claster_size = list()
        for c in claster_list[t-1]:
            claster_size.append(len(c))
        ic = np.argsort(claster_size)[::-1]
        nc = np.sort(claster_size)[::-1]
        #print(f"__{nc}")
        for c in ic:
            claster_list[t-1].append(claster_list[t-1][c])
        for c in ic:
            del claster_list[t-1][0]
        #print(f"{type(claster_list[t])}")
        #print(f"{claster_list[t]}")
        idx = [claster_list[t-1][l][i] for l in range(len(claster_list[t-1])) for i in range(len(claster_list[t-1][l]))]
        #print(idx)
        #print(len(idx))
        print(f"{n_train}/{n_labeled}")
        status[idx[:n_train]] = 2

    status_name = ['background', 'test', 'training']

    return status, status_name

def labeled_unlabeled_test_split(prop_train_l, status, unlabeled_type = 'from_train', coh_threshold=0.5, seed_l=None, seed_u=None, recategorize_rule=recategorize17to10_csv, gt_gic=True):
    """
    prop_train_l
    - labeled data proportion to train data. (NOTE: proportion to annotated data is prop_train * prop_train_l)

    unlabeled_type
    - "from_train" :            selected from the group of status == 2 (training data)
    - "from_spatial" : selected from remained annotated data after selecting labeled, by neighboring and close-spectrum.
    - "from_other" :            selected from remained annotated data after selecting labeled, randomly

    seed_l : start_idx which define to select labeled from train data

    seed_u : shuffle seed for selecting unlabeled from annotated data

    output l_u_t_status, l_u_t_status_name
    - l_u_t_status_name = ['background', 'test', 'unlabeled', 'labeled']
    - l_u_t_status : statuses for each instance in `l_u_t_status_name`
    """

    rg1 = default_rng(seed_l)
    rg2 = default_rng(seed_u)

    def xy2idx(x,y):
        ly = np.max(cordinates[:,1])+1
        return y + x*ly

    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_rule,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : gt_gic,
    }
    pines = IP.load(**IP_conf)

    labels = sorted(list(set(pines.target)))
    status = np.array(status)
    l_u_t_status = np.zeros_like(status) # labeledの初期化．0は，background(non-annotated)を表す
    for t in labels:
        idx_train = list()
        idx_test = list()
        for i,ti in enumerate(pines.target):
            if ti == t:            # targetが t である．
                if status[i] == 2: # 決められたtrainingに入っている
                    idx_train.append(i)
                elif status[i] == 1:
                    idx_test.append(i)
        l_u_t_status[idx_test] = 1            # targetが t で，statusがtest　を 1[test]に初期化
        l_u_t_status[idx_train] = 2           # targetが t で，statusがtraining　を2[unlabeled]に初期化
        n_train = len(idx_train)
        n_labeled = int(np.ceil(prop_train_l * n_train))
        n_unlabeled = n_train - n_labeled
        st = int(np.floor(rg1.uniform(low=0, high=n_unlabeled)))
        l_u_t_status[idx_train[st:st+n_labeled]] = 3 # 2[unlabeled]から継続して選んだものを3「labeled」とする．
        if unlabeled_type != 'from_train':
            # status = 1 「trainingデータからlabeledを抽出した残り」 を test に統合
            l_u_t_status[l_u_t_status == 2] = 1
            idx_test = np.where(l_u_t_status == 1)[0]
            idx_test = rg2.permutation(idx_test)
            if unlabeled_type == 'from_annot':
                l_u_t_status[idx_test[:n_unlabeled]] = 2
            elif unlabeled_type == 'from_spatialneighbor':
                for i in idx_test:
                    xi, yi = pines.cordinates[i]
                    i_fourneighbor = [xy2idx(xi-1,yi), xy2idx(xi,yi-1), xy2idx(xi, yi+1), xy2idx(xi+1,yi)]
                    i_eightneighbor = i_fourneighbor + [xy2idx(xi-1,yi-1), xy2idx(xi-1,yi+1), xy2idx(xi+1, yi-1), xy2idx(xi+1, yi+1)]
                    i_neighbor = np.array(i_fourneighbor) ## <--- SET four or eight
                    i_neighbor = i_neighbor[i_neighbor>0] # 実際に存在するインデクスに限定
                    min_coherence = 1e+7
                    for j in i_neighbor:
                        coh = pines.features[i] @ pines.features[j].T
                        coh /= pines.features[j] @ pines.features[j].T
                        if coh < min_coherence: min_coherence = coh
                    if min_coherence > coh_threshold:
                        l_u_t_status[i] = 2

    l_u_t_status_name = ['background', 'test', 'unlabeled', 'labeled']
    return l_u_t_status, l_u_t_status_name

def colored_map(ax,target,cordinates,recategorize_rule=recategorize17to10_csv,gt_gic=True):
    """targetの色分け地図を描画

    Args:
        target (_type_): _description_
        cordinates (_type_): _description_
    """
    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_rule,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : gt_gic,
    }
    pines = IP.load(**IP_conf)

    mapcordinates_df = pd.DataFrame([(x, y) for x in range(0,145) for y in range(0,145)],columns=['#x','#y'])

    hex_df = pd.DataFrame(pines.hex_names[target],columns=['hex-color'])
    cordinates_df = pd.DataFrame(cordinates,columns=['#x','#y'])
    df = pd.concat([cordinates_df,hex_df],axis=1)
    df = pd.merge(mapcordinates_df,df,on=['#x','#y'],how='left')

    l_hex_names = np.array([c+'40' for c in pines.hex_names])
    l_id = pines.target>0
    l_target = pines.target[l_id]
    l_cordinates = pines.cordinates[l_id]
    l_hex_df = pd.DataFrame(l_hex_names[l_target],columns=['hex-color'])
    l_cordinates_df = pd.DataFrame(l_cordinates,columns=['#x','#y'])
    l_df = pd.concat([l_cordinates_df,l_hex_df],axis=1)
    l_df = pd.merge(mapcordinates_df,l_df,on=['#x','#y'],how='left')

    id = df['hex-color'].isna()
    df[id] = l_df[id]
    df = df.fillna('#ffffff')

    ax.imshow(colors.to_rgba_array(df['hex-color'].values).reshape([145,145,4]))

    for i,c in enumerate(pines.hex_names):
        ax.scatter([],[],c=c,marker='s',label=pines.target_names[i])
    # legend 付けたいが，，，，ax.plotで作ってないので，どうするんだろ？
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
