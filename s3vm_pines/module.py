import indianpines as IP
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
import pandas as pd
from importlib import resources

recategorize_csv = resources.files('s3vm_pines')/'recategolize'/'recategorize17to10.csv'

def make_traininigSet(Area=2,NumWanted=5,seed=0):
    """
    データセットに対する
    - unlabeled(rest) = 0
    - labeled(training) = 1
    - test(test) = 2
    - unused(background) = 3
    のタグ付けリスト(status)とタグ名(status_name)を返す．
    labeled(1)とtest(2)は隣接Area*Areaで同一カテゴリとなっている領域から作成される．

    * 基本的には一度作ってcsv保存とかしておけばいいのかも

    Input:
    - Area (int): labeledとtestに設定する同一カテゴリ連続隣接矩形領域の辺長ピクセル数
    - NumWanted (int): labeledとtestそれぞれに欲しい連続領域の個数
    - seed (int): 乱数シード

    Output:
    - status (np.array(SampleSize,)): タグ番号リスト
    - status_name (str List(4,)): タグ番号順に並べたタグ名の対応リスト
    """

    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_csv,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : True,
    }
    pines = IP.load(**IP_conf)

    # Area x Area　の同一カテゴリの領域をトレーニングエリアとする．
    halfArea=Area//2
    rng = np.random.default_rng(seed)

    ly = np.max(pines.cordinates[:,0])+1
    lx = np.max(pines.cordinates[:,1])+1
    cnt_tra = np.zeros(10)
    cnt_tes = np.zeros(10)
    i_tra = []
    i_tes = []
    for y in range(halfArea,ly-halfArea):
        for x in range(halfArea,lx-halfArea):
            I = []
            UI = []
            LI = []
            for dy in range(-halfArea,Area-halfArea):
                for dx in range(-halfArea,Area-halfArea):
                    I.append((y+dy)*lx+(x+dx))
            t = pines.target[I[0]]
            for dx in range(-halfArea-1,Area-halfArea-1):
                UI.append(I[0]-Area+dx)

            for dy in range(-halfArea,Area-halfArea):
                LI.append(I[0]-1+(dy+halfArea)*lx)

            if t != 0 and np.all(pines.target[I] == t):
                if np.any(pines.target[UI] != t) and np.any(pines.target[LI] != t):
                    if rng.binomial(1,0.5):
                        if cnt_tra[t] < NumWanted:
                            for i in I:
                                i_tra.append([i,t])
                            cnt_tra[t] = cnt_tra[t]+1
                    else:
                        if cnt_tes[t] < NumWanted:
                            for i in I:
                                i_tes.append([i,t])
                            cnt_tes[t] = cnt_tes[t]+1
    i_tra = np.array(i_tra)
    i_tes = np.array(i_tes)


    # unused(background) = 3, labeled(training) = 1,
    # test(test) = 2, unlabeled(rest) = 0
    status = np.zeros((lx*ly,))
    for i,t in i_tra:
        status[i] = 1
    for i,t in i_tes:
        status[i] = 2
    for i,t in enumerate(pines.target):
        if t == 0:
            status[i] = 3

    status_name = ['unlabeled','labeled','test','background']

    return status, status_name

def colored_map(ax,target,cordinates):
    """targetの色分け地図を描画

    Args:
        target (_type_): _description_
        cordinates (_type_): _description_
    """
    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule" : recategorize_csv,
        "exclude_WaterAbsorptionChannels" : True,
        "gt_gic" : True,
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
