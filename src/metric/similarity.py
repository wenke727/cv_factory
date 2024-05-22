# Ref: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/deploy/pipeline/pphuman/mtmct.py
import numpy as np


def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        np.power(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
        + np.power(y, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    )
    distmat -= np.dot(2 * x, y.T)

    return distmat


def cosine_similarity(x, y, eps=1e-12):
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n = np.linalg.norm(x, axis=1, keepdims=True)
    y_n = np.linalg.norm(y, axis=1, keepdims=True)
    x_norm = x / np.maximum(x_n, eps * np.ones_like(x_n))
    y_norm = y / np.maximum(y_n, eps * np.ones_like(y_n))
    sim_mt = np.dot(x_norm, y_norm.T)

    return sim_mt


def get_cosine(x, y, eps=1e-12):
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behavior to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)

    return sim_mt


def get_dist_mat(x, y, func_name="euclidean"):
    if func_name == "cosine":
        dist_mat = get_cosine(x, y)
    elif func_name == "euclidean":
        dist_mat = get_euclidean(x, y)
    print("Using {} as distance function during evaluation".format(func_name))

    return dist_mat


# MOT 相关
def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][1] == cid_tids[j][1]:
                st_mask[i, j] = 0.
    return st_mask


def calculate_intercamera_similarity_matrix(cid_tid_dict, cid_tids):
    # get_sim_matrix_new
    # Note: camera independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    #compute distmat
    distmat = get_dist_mat(q_arr, g_arr, func_name="cosine")

    #mask the element which belongs to same video
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    sim_matrix = distmat * st_mask
    np.fill_diagonal(sim_matrix, 0.)
    return 1. - sim_matrix
