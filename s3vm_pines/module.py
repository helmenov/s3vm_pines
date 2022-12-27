from importlib import resources

import IndianPines as IP
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy.random import default_rng
import logging

recategorize17to10_csv = (
    resources.files("s3vm_pines") / "recategolize" / "recategorize17to10.csv"
)


def xy2idx(x_coord: int, y_coord: int, Lx: int, Ly: int) -> int:
    """Given (x,y) and (Lx,Ly), then return image pixel index

    Args:
        x_coord, y_coord: int. x-axises and y-axises. [[0,0]] means Left-Upper side coordinate.
        Lx and Ly: int. image width and height in pixel
    Return:
        index: image pixel index (y_coord * Lx + x_coord)
    """
    assert Lx * Ly > 0
    assert x_coord < Lx
    assert y_coord < Ly

    idx = int(y_coord * Lx + x_coord)

    return idx


def hilbert_index(lx: int, ly: int, morton: bool=False):
    """
    generate Hilbert scan ordered indices

    Args:
        lx and ly: (int) image width and height in pixel
        morton: (bool) morton indices
    Return:
        indices: NDArray (2^p*2^p, 1). Hilbert indices
    """
    p = int(np.ceil(np.log2(np.max([lx,ly]))))

    def p_hilbert_index(p, morton):
        if p == 1:
            if morton == False:
                coord = [[1,0],[0,0],[0,1],[1,1]]
            else:
                coord = [[1,0],[0,0],[1,1],[0,1]]
            return np.array(coord, dtype=int)
        else:
            bias = 2 * p_hilbert_index(p=p-1, morton=morton)
            for i, b in enumerate(bias):
                c = p_hilbert_index(p=1,morton=morton) + np.tile(b,(4,1))
                if morton == False:
                    if i == 0:
                        c[1], c[3] = c[3],c[1]
                    elif i == 2:
                        c[0], c[2] = c[2],c[0]
                if i==0:
                    coord = c
                else:
                    coord = np.r_[coord,c]
            return coord
    coord = p_hilbert_index(p=p,morton=morton)
    hilbert_indices: list[int] = list()
    for r,c in coord:
        if r<ly and c<lx:
            hilbert_indices.append(int(r*lx+c))
    return np.array(hilbert_indices, dtype=int)

def zigzag_index(lx: int, ly: int) :
    """Given (lx,ly), then return zigzag indices which length is lx*ly

    Args:
        lx and ly: int. image width and height in pixel
    Return:
        indices: NDArray (lx*ly,1). zigzag indices.
    """
    assert lx * ly > 0
    zigzag_indices: list[int] = list()
    for ix in range(lx + ly - 1):
        if ix < lx:
            nix = ix
            niy = 0
        else:
            nix = lx - 1
            niy = ix - lx + 1
        # print(f'({nix},{niy})={niy*lx+nix}')
        zigzag_indices.append(niy * lx + nix)
        while 0 <= nix - 1 and niy + 1 < ly:
            nix = nix - 1
            niy = niy + 1
            zigzag_indices.append(niy * lx + nix)
            # print(f'({nix},{niy})={niy*lx+nix}')
    np_zigzag_indices = np.array(zigzag_indices, dtype=int)
    return np_zigzag_indices


def scanner_sortindices(indices, scanner_indices):
    """sort pixel indices to zigzag_indices order

    Args:
        indices: image pixel indices
        scanner_indices: returned output of func:zigzag_index or hilbert_index
    Return:
        out_indices: sorted indices
    """
    out_indices = list()
    for idx in scanner_indices:
        for idx2 in indices:
            if idx == idx2:
                out_indices.append(idx2)
            else:
                continue
    out_indices = np.array(out_indices, dtype=int)
    return out_indices

def train_test_split(
    p_train=1,
    recategorize_rule=recategorize17to10_csv,
    gt_gic=True,
    seed_train=0,
    n_train_equal=True
):
    """split whole data to 'train' and 'test'
    すべての座標上のデータをカテゴリごとに領域分割し，
    広い領域から連続的に，trainを半分先取り，残りをtestとする

    Args:
        p_train: 0<=p_train<=1. proportion to training instance in each class. default is 1 (all labeled is trained)
        recategorize_rule: default is recategorize17to10.csv
        gt_gic: default is True
        seed_train: default is 0
        n_train_equal: (bool) if True, all classes has same number of instances (= N*p_train/n_class, for whole labeled number is N).
                              if False, (= n*p_train, for each class labeled number is n). default is True.

    Returns:
        train_test_status: status labels NDArray
        train_test_status_name: ["background","test","training"]
    """
    if seed_train > 0:
        rg_train = default_rng(seed_train)

    IP_conf = {
        "include_background": True,
        "recategorize_rule": recategorize_rule,
        "gt_gic": gt_gic,
    }
    _pines = IP.load(**IP_conf)

    Lx = np.amax(_pines.coordinates[:, 0]) + 1
    Ly = np.amax(_pines.coordinates[:, 1]) + 1

    n_class = _pines.target_names.shape[0]
    cluster_list = list()
    ####
    print(f"Now clustering for each target category")
    ####
    scanner = hilbert_index(Lx, Ly, morton=True)
    for t in range(1, n_class):
        ####
        print(f"\t{t}/{n_class}: {_pines.target_names[t]}", end="")
        ####
        labeled_cluster = np.zeros_like(_pines.target)
        LookupTable = list()
        label = 0
        LookupTable.append(label)
        for xi, yi in _pines.coordinates[_pines.target == t]:
            c_Neighbors = list()
            # UpperLeft
            if (
                0 <= xi - 1 < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi - 1, yi - 1, Lx, Ly)] == t
            ):
                c_Neighbors.append(
                    labeled_cluster[xy2idx(xi - 1, yi - 1, Lx, Ly)]
                )
            # Upper
            if (
                0 <= xi < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi, yi - 1, Lx, Ly)] == t
            ):
                c_Neighbors.append(labeled_cluster[xy2idx(xi, yi - 1, Lx, Ly)])
            # UpperRight
            if (
                0 <= xi + 1 < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi + 1, yi - 1, Lx, Ly)] == t
            ):
                c_Neighbors.append(
                    labeled_cluster[xy2idx(xi + 1, yi - 1, Lx, Ly)]
                )
            # Left
            if (
                0 <= xi - 1 < Lx
                and 0 <= yi < Ly
                and _pines.target[xy2idx(xi - 1, yi, Lx, Ly)] == t
            ):
                c_Neighbors.append(labeled_cluster[xy2idx(xi - 1, yi, Lx, Ly)])
            ## UnderLeft
            # if 0<= xi - 1 < Lx and 0<= yi + 1 < Ly and _pines.target[xy2idx(xi-1,yi+1,Lx,Ly)] == t:
            #    c_Neighbors.append(labeled_cluster[xy2idx(xi - 1, yi + 1, Lx, Ly)])
            c_Neighbors = np.array(c_Neighbors)
            # print(c_Neighbors)
            if np.all(c_Neighbors == 0):
                label += 1
                LookupTable.append(label)
                labeled_cluster[xy2idx(xi, yi, Lx, Ly)] = label
                # print(f"({xi},{yi}) is labeled {label} [A]")
            else:
                if len(c_Neighbors[c_Neighbors > 0]) > 0:
                    labeled_cluster[xy2idx(xi, yi, Lx, Ly)] = min(
                        c_Neighbors[c_Neighbors > 0]
                    )
                    # print(f"({xi},{yi}) is labeled {min(c_Neighbors[c_Neighbors>0])}[B]")
                    # print(f"labeled_cluster[{xy2idx(xi,yi,Lx,Ly)}] = {labeled_cluster[xy2idx(xi,yi,Lx,Ly)]}")
                    for c_other in c_Neighbors[
                        c_Neighbors > labeled_cluster[xy2idx(xi, yi, Lx, Ly)]
                    ]:
                        # print(f"LooupTabel[{c_other}] is {labeled_cluster[xy2idx(xi,yi,Lx,Ly)]}")
                        LookupTable[c_other] = labeled_cluster[
                            xy2idx(xi, yi, Lx, Ly)
                        ]
            # print("---")

        # LookupTableを降順に見て，使われなかったラベルを統合
        L = len(LookupTable)
        for i, j in enumerate(LookupTable[::-1]):
            if j != L - i - 1:
                labeled_cluster[labeled_cluster == L - i - 1] = j

        # indexlistを作る
        cluster_list_t = list()
        cluster_size = list()
        for c in range(1, max(labeled_cluster) + 1):
            cluster_size.append(len(labeled_cluster[labeled_cluster == c]))
        cluster_size = np.array(cluster_size)
        cluster_size_idx = np.argsort(cluster_size)[::-1]
        for c in range(0, max(labeled_cluster)):
            list_idx = [
                idx
                for idx in range(len(labeled_cluster))
                if labeled_cluster[idx] == cluster_size_idx[c] + 1
            ]
            list_idx = scanner_sortindices(list_idx, scanner)
            if len(list_idx) > 0:
                cluster_list_t.append(list_idx)
        # print(len(cluster[cluster == c]))

        # for c in range(len(cluster_list_t)):
        #    print(cluster_list_t[c])
        print(
            f"\t n_clusters:{len(cluster_list_t)}, maxsize={len(cluster_list_t[0])}, minsize={len(cluster_list_t[-1])}"
        )

        cluster_list.append(cluster_list_t)
        # print('='*10)

    # cluster_list[t] : targetのクラスター番号リスト
    # cluster_list[t][l] : targetのl番目のクラスターのインデクスリスト

    ####
    logging.info("Now sampling train and test data")
    ####
    # print(zigzag)
    n_train = _pines.target[_pines.target>0].shape[0]
    train_test_status = np.zeros_like(_pines.target)  # 'background'
    for t in range(1, n_class):
        logging.info(f"\t{t}/{n_class}:{_pines.target_names[t]}")
        idx = _pines.target == t
        train_test_status[idx] = 1  # 'test'
        n_gt_t = _pines.target[idx].shape[0]
        # print(n_labeled)
        if n_train_equal == False:
            n_train_t = int(np.ceil(p_train * n_gt_t))
        else:
            n_train_t = int(np.ceil(n_train*p_train/n_class))
        if n_train_t > n_gt_t:
            n_train_t = n_gt_t
        n_test_t = n_gt_t - n_train_t

        # cluster_listのサイズ降順ソート
        cluster_size = list()
        for c in cluster_list[t - 1]:
            cluster_size.append(len(c))
        ic = np.argsort(cluster_size)[::-1]
        nc = np.sort(cluster_size)[::-1]
        # print(f"__{nc}")
        for c in ic:
            cluster_list[t - 1].append(cluster_list[t - 1][c])
        for c in ic:
            del cluster_list[t - 1][0]
        # print(f"{type(cluster_list[t])}")
        # print(f"{cluster_list[t]}")

        idx = [
            cluster_list[t - 1][l][i]
            for l in range(len(cluster_list[t - 1]))
            for i in range(len(cluster_list[t - 1][l]))
        ]

        # st = 抽出の取り始めindex
        if seed_train > 0:
            st = int(np.floor(rg_train.uniform(low=0, high=n_gt_t - n_test_t)))
        elif seed_train == 0:
            st = 0
        else:
            st = len(idx) - n_train_t
        train_test_status[idx[st : st + n_train_t]] = 2  # 'train'

    train_test_status_name = ["background", "test", "training"]

    return train_test_status, train_test_status_name


def labeled_unlabeled_sample(
    p_labeled:float,
    p_unlabeled:float,
    train_test_status,
    recategorize_rule=recategorize17to10_csv,
    gt_gic:bool=True,
    unlabeled_neighbor_labeled:bool=False,
    seed_labeled:int=0,
    seed_unlabeled:int=0,
):
    """sample labeled and unlabeled data from train data in the each proportion.

    Args:
        p_labeled (float): 0 < p_labeled <= 1, proportion of the number of labeled with respect to the number of train data.
        p_unlabeled (float): 0 < p_labeled*p_unlabeled <= 1, proportion of the number of unlabeled with respect to the number of train data.
        train_test_status (NDArray): referenced 'train_test_status' labels which 'train_test_split' method yields.
        recategorize_rule: Defaults to recategorize17to10_csv
        gt_gic: Defaults to True
        unlabeled_neighbor_labeled (bool): Trueの場合，ラベルなしデータを，空間上でラベル付きデータに隣りあっているデータ群から採集. Falseの場合にはランダムに採集．Defaults to False.
        seed_labeled (int, optional): seed for random heading of sampling labeled. Defaults to None.
        seed_unlabeled (int, optional): seed for random sampling unlabeled. Defaults to None.

    Return:
        status: status labels NDArray
        status_names: ["background", "test", "training_rest", "labeled", "unlabeled"]
    """
    n_whole = train_test_status.shape[0]
    n_train = train_test_status[train_test_status == 2].shape[0]
    n_labeled = int(np.ceil(p_labeled * n_train))
    n_unlabeled = p_unlabeled*n_labeled
    if n_unlabeled > n_train-n_labeled:
        n_unlabeled = n_train-n_labeled
        print(f'p_unlabeled: {p_unlabeled} -> {n_unlabeled/n_labeled}')
        p_unlabeled = n_unlabeled/n_labeled
    logging.info(f"{n_whole=},{n_train=},{n_labeled=},{n_unlabeled=}")
    assert n_labeled + n_unlabeled <= n_train

    if seed_labeled > 0:
        rg_l = default_rng(seed_labeled)
    rg_u = default_rng(seed_unlabeled)

    IP_conf = {
        "include_background": True,
        "recategorize_rule": recategorize_rule,
        "gt_gic": gt_gic,
    }
    _pines = IP.load(**IP_conf)

    Lx = np.amax(_pines.coordinates[:, 0]) + 1
    Ly = np.amax(_pines.coordinates[:, 1]) + 1

    n_class = _pines.target_names.shape[0]
    cluster_list = list()
    ####
    logging.info(f"Now clustering for each target category")
    ####
    scanner = hilbert_index(Lx, Ly, morton=True)
    for t in range(1, n_class):
        ####
        logging.info(f"\t{t}/{n_class}: {_pines.target_names[t]}")
        ####
        labeled_cluster = np.zeros_like(_pines.target)
        LookupTable = list()
        label = 0
        LookupTable.append(label)
        for xi, yi in _pines.coordinates[
            (_pines.target == t) & (train_test_status == 2)
        ]:
            c_Neighbors = list()
            # UpperLeft
            if (
                0 <= xi - 1 < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi - 1, yi - 1, Lx, Ly)] == t
                and train_test_status[xy2idx(xi - 1, yi - 1, Lx, Ly)] == 2
            ):
                c_Neighbors.append(
                    labeled_cluster[xy2idx(xi - 1, yi - 1, Lx, Ly)]
                )
            # Upper
            if (
                0 <= xi < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi, yi - 1, Lx, Ly)] == t
                and train_test_status[xy2idx(xi, yi - 1, Lx, Ly)] == 2
            ):
                c_Neighbors.append(labeled_cluster[xy2idx(xi, yi - 1, Lx, Ly)])
            # UpperRight
            if (
                0 <= xi + 1 < Lx
                and 0 <= yi - 1 < Ly
                and _pines.target[xy2idx(xi + 1, yi - 1, Lx, Ly)] == t
                and train_test_status[xy2idx(xi + 1, yi - 1, Lx, Ly)] == 2
            ):
                c_Neighbors.append(
                    labeled_cluster[xy2idx(xi + 1, yi - 1, Lx, Ly)]
                )
            # Left
            if (
                0 <= xi - 1 < Lx
                and 0 <= yi < Ly
                and _pines.target[xy2idx(xi - 1, yi, Lx, Ly)] == t
                and train_test_status[xy2idx(xi - 1, yi, Lx, Ly)] == 2
            ):
                c_Neighbors.append(labeled_cluster[xy2idx(xi - 1, yi, Lx, Ly)])
            # UnderLeft
            # if 0<= xi - 1 < Lx and 0<= yi + 1 < Ly and _pines.target[xy2idx(xi-1,yi+1,Lx,Ly)] == t and train_test_status[xy2idx(xi-1,yi+1,Lx,Ly)]==2:
            #    c_Neighbors.append(labeled_cluster[xy2idx(xi - 1, yi + 1, Lx, Ly)])
            c_Neighbors = np.array(c_Neighbors)
            # print(c_Neighbors)
            if np.all(c_Neighbors == 0):
                label += 1
                LookupTable.append(label)
                labeled_cluster[xy2idx(xi, yi, Lx, Ly)] = label
                # print(f"({xi},{yi}) is labeled {label} [A]")
            else:
                if len(c_Neighbors[c_Neighbors > 0]) > 0:
                    labeled_cluster[xy2idx(xi, yi, Lx, Ly)] = min(
                        c_Neighbors[c_Neighbors > 0]
                    )
                    # print(f"({xi},{yi}) is labeled {min(c_Neighbors[c_Neighbors>0])}[B]")
                    # print(f"labeled_cluster[{xy2idx(xi,yi,Lx,Ly)}] = {labeled_cluster[xy2idx(xi,yi,Lx,Ly)]}")
                    for c_other in c_Neighbors[
                        c_Neighbors > labeled_cluster[xy2idx(xi, yi, Lx, Ly)]
                    ]:
                        # print(f"LooupTabel[{c_other}] is {labeled_cluster[xy2idx(xi,yi)]}")
                        LookupTable[c_other] = labeled_cluster[
                            xy2idx(xi, yi, Lx, Ly)
                        ]
            # print("---")

        # LookupTableを降順に見て，使われなかったラベルを統合
        L = len(LookupTable)
        for i, j in enumerate(LookupTable[::-1]):
            if j != L - i - 1:
                labeled_cluster[labeled_cluster == L - i - 1] = j

        # indexlistを作る
        cluster_list_t = list()
        cluster_size = list()
        for c in range(1, max(labeled_cluster) + 1):
            cluster_size.append(len(labeled_cluster[labeled_cluster == c]))
        cluster_size = np.array(cluster_size)
        cluster_size_idx = np.argsort(cluster_size)[::-1]
        for c in range(0, max(labeled_cluster)):
            list_idx = [
                idx
                for idx in range(len(labeled_cluster))
                if labeled_cluster[idx] == cluster_size_idx[c] + 1
            ]
            list_idx = scanner_sortindices(list_idx, scanner)
            if len(list_idx) > 0:
                cluster_list_t.append(list_idx)
            # print(len(cluster[cluster == c]))
        # print(cluster_list_t)
        cluster_list.append(cluster_list_t)
        # print('='*10)

    # cluster_list[t] : targetのクラスター番号リスト
    # cluster_list[t][l] : targetのl番目のクラスターのインデクスリスト

    ####
    logging.info("Now sampling labeled and unlabeled data")
    ####
    status = np.zeros_like(_pines.target) # background
    for t in range(1, n_class):
        logging.info(f"\t{t}/{n_class}:{_pines.target_names[t]}")
        idx = _pines.target == t
        status[idx] = 1 # relabel background -> test
        idx = (_pines.target == t) & (train_test_status == 2)
        status[idx] = 2 # relabel test -> training_rest
        n_train_t = _pines.target[idx].shape[0]
        # print(n_labeled)
        n_labeled_t = int(np.ceil(p_labeled * n_train_t))
        n_unlabeled_t = int(np.ceil(p_unlabeled * n_train_t))
        cluster_size = list()
        for c in cluster_list[t - 1]:
            cluster_size.append(len(c))
        ic = np.argsort(cluster_size)[::-1]
        nc = np.sort(cluster_size)[::-1]
        # print(f"__{nc}")
        for c in ic:
            cluster_list[t - 1].append(cluster_list[t - 1][c])
        for c in ic:
            del cluster_list[t - 1][0]
        # print(f"{type(cluster_list[t])}")
        # print(f"{cluster_list[t]}")
        idx = [
            cluster_list[t - 1][l][i]
            for l in range(len(cluster_list[t - 1]))
            for i in range(len(cluster_list[t - 1][l]))
        ]
        # print(idx)
        # print(len(idx))
        # idx = scanner_sortindices(idx, scanner)

        if seed_labeled > 0:
            st = int(
                np.floor(rg_l.uniform(low=0, high=n_train_t - n_unlabeled_t))
            )
        elif seed_labeled == 0:
            st = 0
        else:
            st = len(idx) - n_labeled_t
        status[idx[st : st + n_labeled_t]] = 3 # (3)labeled

    idx_train_rest = np.where((status == 2))[0]
    idx_train_rest = rg_u.permutation(idx_train_rest)
    n_train_rest = len(idx_train_rest)
    n_labeled = len(status[status==3])
    n_unlabeled = min(int(np.ceil(n_labeled*p_unlabeled)), len(idx_train_rest))
    if unlabeled_neighbor_labeled == True:
        cnt_u = 0
        for i in idx_train_rest:
            iUL = xy2idx(
                _pines.coordinate[i, 0] - 1,
                _pines.coordinate[i, 1] - 1,
                Lx,
                Ly,
            )
            iUC = xy2idx(
                _pines.coordinate[i, 0],
                _pines.coordinate[i, 1] - 1,
                Lx,
                Ly,
            )
            iUR = xy2idx(
                _pines.coordinate[i, 0] + 1,
                _pines,
                coordinate[i, 1] - 1,
                Lx,
                Ly,
            )
            iCL = xy2idx(
                _pines.coordinate[i, 0] - 1,
                _pines.coordinate[i, 1],
                Lx,
                Ly,
            )
            iCR = xy2idx(
                _pines.coordinate[i, 0] + 1,
                _pines.coordinate[i, 1],
                Lx,
                Ly,
            )
            iDL = xy2idx(
                _pines.coordinate[i, 0] - 1,
                _pines.coordinate[i, 1] + 1,
                Lx,
                Ly,
            )
            iDC = xy2idx(
                _pines.coordinate[i, 0],
                _pines.coordinate[i, 1] + 1,
                Lx,
                Ly,
            )
            iDR = xy2idx(
                _pines.coordinate[i, 0] + 1,
                _pines.coordinate[i, 1] + 1,
                Lx,
                Ly,
            )
            if (
                status[iUL] == 3
                or status[iUC] == 3
                or status[iUR] == 3
                or status[iCL] == 3
                or status[iCR] == 3
                or status[iDL] == 3
                or status[iDC] == 3
                or status[iDR] == 3
            ):
                status[i] = 4 # (4)unlabeled
                cnt_u += 1
                if cnt_u < n_unlabeled:
                    continue
    else:
        status[idx_train_rest[:n_unlabeled]] = 4

    status_name = [
        "background",
        "test",
        "training_rest",
        "labeled",
        "unlabeled",
    ]

    return status, status_name


def train_test_split2(
    p_train=0.5, recategorize_rule=recategorize17to10_csv, gt_gic=True
):
    """
    アノテーションデータを train:test = p:1-p に分割する．
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
        "recategorize_rule": recategorize_rule,
        "exclude_WaterAbsorptionChannels": True,
        "gt_gic": gt_gic,
    }
    pines = IP.load(**IP_conf)
    n_class = pines.target_names.shape[0]

    Lx = np.max(pines.coordinates[:, 0]) + 1
    Ly = np.max(pines.coordinates[:, 1]) + 1

    cluster_list = list()
    zigzag = zigzag_index(Lx, Ly)
    for t in range(1, n_class):
        cluster = np.zeros_like(pines.target)
        LookupTable = list()
        label = 0
        LookupTable.append(label)
        for xi, yi in pines.coordinates[pines.target == t]:
            c_Neighbors = list()
            # UpperLeft
            if 0 <= xi - 1 < Lx and 0 <= yi - 1 < Ly:
                c_Neighbors.append(cluster[xy2idx(xi - 1, yi - 1, Lx, Ly)])
            # Upper
            if 0 <= xi < Lx and 0 <= yi - 1 < Ly:
                c_Neighbors.append(cluster[xy2idx(xi, yi - 1, Lx, Ly)])
            # UpperRight
            if 0 <= xi + 1 < Lx and 0 <= yi - 1 < Ly:
                c_Neighbors.append(cluster[xy2idx(xi + 1, yi - 1, Lx, Ly)])
            # Left
            if 0 <= xi - 1 < Lx and 0 <= yi < Ly:
                c_Neighbors.append(cluster[xy2idx(xi - 1, yi, Lx, Ly)])
            ## UnderLeft
            # if 0<= xi - 1 < Lx and 0 <= yi + 1 < Ly:
            #    c_Neighbors.append(cluster[xy2idx(xi - 1, yi + 1, Lx, Ly)])
            c_Neighbors = np.array(c_Neighbors)
            # print(c_Neighbors)
            if np.all(c_Neighbors == 0):
                label += 1
                LookupTable.append(label)
                cluster[xy2idx(xi, yi, Lx, Ly)] = label
                # print(f"({xi},{yi}) is labeled {label} [A]")
            else:
                if len(c_Neighbors[c_Neighbors > 0]) > 0:
                    cluster[xy2idx(xi, yi, Lx, Ly)] = min(
                        c_Neighbors[c_Neighbors > 0]
                    )
                    # print(f"({xi},{yi}) is labeled {min(c_Neighbors[c_Neighbors>0])}[B]")
                    # print(f"cluster[{xy2idx(xi,yi, Lx, Ly)}] = {cluster[xy2idx(xi,yi, Lx, Ly)]}")
                    for c_other in c_Neighbors[
                        c_Neighbors > cluster[xy2idx(xi, yi, Lx, Ly)]
                    ]:
                        LookupTable[c_other] = cluster[xy2idx(xi, yi, Lx, Ly)]
            # print("---")

        # LookupTableを降順に見て，使われなかったラベルを統合
        L = len(LookupTable)
        for i, j in enumerate(LookupTable[::-1]):
            if j != L - i - 1:
                cluster[cluster == L - i - 1] = j

        # indexlistを作る
        cluster_list_t = list()
        for c in range(1, max(cluster) + 1):
            list_idx = [
                idx for idx in range(len(cluster)) if cluster[idx] == c
            ]
            list_idx = zigzag_sortindices(list_idx, zigzag)
            cluster_list_t.append(list_idx)
            # print(len(cluster[cluster == c]))
        # print(cluster_list_t)
        cluster_list.append(cluster_list_t)
        # print('='*10)

    # cluster_list[t] : targetのクラスター番号リスト
    # cluster_list[t][l] : targetのl番目のクラスターのインデクスリスト

    status = np.zeros_like(pines.target)
    for t in range(1, n_class):
        idx = pines.target == t
        status[idx] = 1
        n_labeled = pines.target[pines.target == t].shape[0]
        # print(n_labeled)
        n_train = int(np.ceil(p_train * n_labeled))
        cluster_size = list()
        for c in cluster_list[t - 1]:
            cluster_size.append(len(c))
        ic = np.argsort(cluster_size)[::-1]
        nc = np.sort(cluster_size)[::-1]
        # print(f"__{nc}")
        for c in ic:
            cluster_list[t - 1].append(cluster_list[t - 1][c])
        for c in ic:
            del cluster_list[t - 1][0]
        # print(f"{type(cluster_list[t])}")
        # print(f"{cluster_list[t]}")
        idx = [
            cluster_list[t - 1][l][i]
            for l in range(len(cluster_list[t - 1]))
            for i in range(len(cluster_list[t - 1][l]))
        ]
        # print(idx)
        # print(len(idx))
        logging.info(f"{n_train}/{n_labeled}")
        status[idx[:n_train]] = 2

    status_name = ["background", "test", "training"]

    return status, status_name


def labeled_unlabeled_test_split(
    prop_train_l,
    status,
    unlabeled_type="from_train",
    coh_threshold=0.5,
    seed_l=None,
    seed_u=None,
    recategorize_rule=recategorize17to10_csv,
    gt_gic=True,
):
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

    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule": recategorize_rule,
        "exclude_WaterAbsorptionChannels": True,
        "gt_gic": gt_gic,
    }
    pines = IP.load(**IP_conf)

    labels = sorted(list(set(pines.target)))
    status = np.array(status)
    l_u_t_status = np.zeros_like(
        status
    )  # labeledの初期化．0は，background(non-annotated)を表す
    for t in labels:
        idx_train = list()
        idx_test = list()
        for i, ti in enumerate(pines.target):
            if ti == t:  # targetが t である．
                if status[i] == 2:  # 決められたtrainingに入っている
                    idx_train.append(i)
                elif status[i] == 1:
                    idx_test.append(i)
        l_u_t_status[idx_test] = 1  # targetが t で，statusがtest　を 1[test]に初期化
        l_u_t_status[
            idx_train
        ] = 2  # targetが t で，statusがtraining　を2[unlabeled]に初期化
        n_train = len(idx_train)
        n_labeled = int(np.ceil(prop_train_l * n_train))
        n_unlabeled = n_train - n_labeled
        st = int(np.floor(rg1.uniform(low=0, high=n_unlabeled)))
        l_u_t_status[
            idx_train[st : st + n_labeled]
        ] = 3  # 2[unlabeled]から継続して選んだものを3「labeled」とする．
        if unlabeled_type != "from_train":
            # status = 1 「trainingデータからlabeledを抽出した残り」 を test に統合
            l_u_t_status[l_u_t_status == 2] = 1
            idx_test = np.where(l_u_t_status == 1)[0]
            idx_test = rg2.permutation(idx_test)
            if unlabeled_type == "from_annot":
                l_u_t_status[idx_test[:n_unlabeled]] = 2
            elif unlabeled_type == "from_spatialneighbor":
                for i in idx_test:
                    xi, yi = pines.coordinates[i]
                    i_fourneighbor = [
                        xy2idx(xi - 1, yi, Lx, Ly),
                        xy2idx(xi, yi - 1, Lx, Ly),
                        xy2idx(xi, yi + 1, Lx, Ly),
                        xy2idx(xi + 1, yi, Lx, Ly),
                    ]
                    i_eightneighbor = i_fourneighbor + [
                        xy2idx(xi - 1, yi - 1, Lx, Ly),
                        xy2idx(xi - 1, yi + 1, Lx, Ly),
                        xy2idx(xi + 1, yi - 1, Lx, Ly),
                        xy2idx(xi + 1, yi + 1, Lx, Ly),
                    ]
                    i_neighbor = np.array(
                        i_fourneighbor
                    )  ## <--- SET four or eight
                    i_neighbor = i_neighbor[i_neighbor > 0]  # 実際に存在するインデクスに限定
                    min_coherence = 1e7
                    for j in i_neighbor:
                        coh = pines.features[i] @ pines.features[j].T
                        coh /= pines.features[j] @ pines.features[j].T
                        if coh < min_coherence:
                            min_coherence = coh
                    if min_coherence > coh_threshold:
                        l_u_t_status[i] = 2

    l_u_t_status_name = ["background", "test", "unlabeled", "labeled"]
    return l_u_t_status, l_u_t_status_name


def colored_map(
    ax,
    target,
    coordinates,
    recategorize_rule=recategorize17to10_csv,
    gt_gic=True,
    with_legend=True
):
    """targetの色分け地図を描画

    Args:
        target (_type_): _description_
        coordinates (_type_): _description_
    """
    IP_conf = {
        "pca": 5,
        "include_background": True,
        "recategorize_rule": recategorize_rule,
        "exclude_WaterAbsorptionChannels": True,
        "gt_gic": gt_gic,
    }
    pines = IP.load(**IP_conf)

    mapcoordinates_df = pd.DataFrame(
        [(x, y) for y in range(0, 145) for x in range(0, 145)],
        columns=["#x", "#y"],
    )

    hex_df = pd.DataFrame(pines.hex_names[target], columns=["hex-color"])
    coordinates_df = pd.DataFrame(coordinates, columns=["#x", "#y"])
    df = pd.concat([coordinates_df, hex_df], axis=1)
    df = pd.merge(mapcoordinates_df, df, on=["#x", "#y"], how="left")

    l_hex_names = np.array([c + "20" for c in pines.hex_names])
    l_id = pines.target > 0
    l_target = pines.target[l_id]
    l_coordinates = pines.coordinates[l_id]
    l_hex_df = pd.DataFrame(l_hex_names[l_target], columns=["hex-color"])
    l_coordinates_df = pd.DataFrame(l_coordinates, columns=["#x", "#y"])
    l_df = pd.concat([l_coordinates_df, l_hex_df], axis=1)
    l_df = pd.merge(mapcoordinates_df, l_df, on=["#x", "#y"], how="left")

    id = df["hex-color"].isna()
    df[id] = l_df[id]  # 主に描画したい対象以外をl_df(whole annotated)で埋める
    df = df.fillna("#ffffff")

    ax.imshow(
        colors.to_rgba_array(df["hex-color"].values).reshape([145, 145, 4])
    )

    if with_legend is True:
        for i, c in enumerate(pines.hex_names):
            ax.scatter([], [], c=c, marker="s", label=pines.target_names[i])
        # legend 付けたいが，，，，ax.plotで作ってないので，どうするんだろ？
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
