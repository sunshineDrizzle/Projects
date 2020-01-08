import itertools
import numpy as np

from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
from community import best_partition
from commontool.algorithm.triangular_mesh import get_n_ring_neighbor
from commontool.algorithm.graph import connectivity_grow


def get_patch_by_crg(vertices, edge_list):
    patches = []
    while vertices:
        seed = vertices.pop()
        patch = connectivity_grow([[seed]], edge_list)[0]
        patches.append(list(patch))
        vertices.difference_update(patch)

    return patches


def get_patch_by_LV(graph):
    partition = best_partition(graph)
    patches_dict = dict()
    for label in partition.values():
        patches_dict[label] = []
    for vtx, label in partition.items():
        patches_dict[label].append(vtx)
    patches = [patches_dict[label] for label in sorted(patches_dict.keys())]

    return patches


def get_roi_pattern(maps, roi, zscore=False, thr=None, bin=False, size_min=0, faces=None, mask=None):
    """

    :param maps: N x M array
        N subjects' hemisphere map
    :param roi: list|1D array
        a collection of vertices of the ROI
    :param zscore: bool
        If True, do z-score on each subject's ROI pattern.
        It will be ignored when 'bin' is True.
    :param thr: float
        A threshold used to cut ROI data before clustering (default: None)
    :param bin: bool
        If True, binarize ROI data according to 'thr'.
        It will be ignored when 'thr' is None.
    :param size_min: non-negative integer
        If is less than or equal to 0, do nothing.
        else, only reserve the patches whose size is larger than 'size_min' after threshold. And
        'faces' must be provided.
        It will be ignored when 'thr' is None or 'mask' is not None.
    :param faces: face_num x 3 array
        It only takes effect when 'size_min' is working.
    :param mask: N x M array
        indices array used to specify valid vertices
        It will be ignored when 'thr' is None

    :return: patterns: N x len(roi) array
        N subjects' ROI pattern
    """
    tmp_maps = maps.copy()
    if thr is not None:
        if mask is not None:
            tmp_maps[np.logical_not(mask)] = thr
        elif size_min > 0:
            patch_maps = np.zeros_like(maps, dtype=np.bool)
            for row in range(patch_maps.shape[0]):
                vertices_thr = set(np.where(maps[row] > thr)[0])
                vertices_thr_roi = vertices_thr.intersection(roi)
                mask = np.zeros(patch_maps.shape[1])
                mask[list(vertices_thr_roi)] = 1
                edge_list = get_n_ring_neighbor(faces, mask=mask)
                patches = get_patch_by_crg(vertices_thr_roi, edge_list)
                for patch in patches:
                    if len(patch) > size_min:
                        patch_maps[row, patch] = True
            tmp_maps[np.logical_not(patch_maps)] = thr
        patterns = tmp_maps[:, roi]
        if bin:
            patterns = (patterns > thr).astype(np.int8)
        else:
            patterns[patterns <= thr] = thr
    else:
        patterns = tmp_maps[:, roi]

    if zscore and not bin:
        patterns = stats.zscore(patterns, 1)

    return patterns


def hac_sklearn(data, n_clusters):

    # do hierarchical clustering on FFA_data by using sklearn
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(data)

    return clustering.labels_


def louvain_community(graph):
    """
    https://python-louvain.readthedocs.io/en/latest/
    https://en.wikipedia.org/wiki/Louvain_Modularity

    :param graph:
    :return:
    """
    partition = best_partition(graph)
    labels = np.zeros(graph.number_of_nodes(), dtype=np.int32)
    for idx, label in partition.items():
        labels[idx] = label + 1

    return [labels]


def girvan_newman_community(graph, max_num):
    """
    https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html#networkx.algorithms.community.centrality.girvan_newman

    :param graph:
    :param max_num:
    :return:
    """

    def most_central_weighted_edge(graph):
        centrality = betweenness(graph, weight='weight')
        return max(centrality, key=centrality.get)

    comp = girvan_newman(graph, most_valuable_edge=most_central_weighted_edge)
    comp_limited = itertools.takewhile(lambda x: len(x) <= max_num, comp)
    labels_list = []
    for communities in comp_limited:
        labels = np.zeros(graph.number_of_nodes(), dtype=np.int32)
        for idx, community in enumerate(communities):
            labels[list(community)] = idx + 1
        labels_list.append(labels)

        print('GN finished: {}/{}'.format(max(labels), max_num))

    return labels_list


def k_means(data, cluster_nums, n_init=10):
    """
    http://scikit-learn.org/stable/modules/clustering.html#k-means

    :param data:
    :param cluster_nums:
    :param n_init:
    :return:
    """
    labels_list = []
    for n_clusters in cluster_nums:
        kmeans = KMeans(n_clusters, random_state=0, n_init=n_init).fit(data)
        labels_list.append(kmeans.labels_ + 1)
        print('KMeans finished: {}'.format(n_clusters))

    return labels_list
