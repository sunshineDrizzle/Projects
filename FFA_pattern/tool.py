import os
import itertools
import numpy as np

from os.path import join as pjoin
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from matplotlib import pyplot as plt
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
from community import best_partition
from commontool.algorithm.triangular_mesh import get_n_ring_neighbor
from commontool.algorithm.graph import connectivity_grow
from commontool.algorithm.plot import VlineMoverPlotter, imshow


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


def gap_stat(data, cluster_nums):
    from gap_statistic import OptimalK

    optimalK = OptimalK()
    return optimalK(data, cluster_array=cluster_nums)


def gap_stat_mine(data, cluster_nums, ref_num=10, cluster_method=None):
    from commontool.algorithm.tool import gap_statistic

    labels_list, Wks, Wks_refs_log_mean, gaps, s, k_selected = \
        gap_statistic(data, cluster_nums, ref_num, cluster_method)

    x = np.arange(len(cluster_nums))
    cluster_nums = np.array(cluster_nums)
    plt.figure()
    plt.plot(x, Wks, 'b.-')
    plt.xlabel('#subgroup')
    plt.ylabel(('W\u2096'))
    if len(x) > 20:
        middle_idx = int(len(x) / 2)
        plt.xticks(x[[0, middle_idx, -1]], cluster_nums[[0, middle_idx, -1]])
    else:
        plt.xticks(x, cluster_nums)

    Wks_log = np.log(Wks)
    plt.figure()
    plt.plot(x, Wks_log, 'b.-')
    plt.plot(x, Wks_refs_log_mean, 'r.-')
    plt.xlabel('#subgroup')
    plt.ylabel('log(W\u2096)')
    if len(x) > 20:
        middle_idx = int(len(x) / 2)
        plt.xticks(x[[0, middle_idx, -1]], cluster_nums[[0, middle_idx, -1]])
    else:
        plt.xticks(x, cluster_nums)

    return labels_list, gaps, s, k_selected


class ClusteringVlineMoverPlotter(VlineMoverPlotter):

    def __init__(self, data, labels_list, clustering_dir, clustering_result_dir,
                 nrows=1, ncols=1, sharex=False, sharey=False,
                 squeese=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
        super(ClusteringVlineMoverPlotter, self).__init__(nrows, ncols, sharex, sharey,
                                                          squeese, subplot_kw, gridspec_kw, **fig_kw)
        self.data = data
        self.labels_list = labels_list
        self.clustering_dir = clustering_dir
        self.clustering_result_dir = clustering_result_dir

    def _on_clicked(self, event):
        if event.button == 3:
            # do something on right click
            if event.inaxes in self.axes:
                vline_mover = self.vline_movers[self.axes == event.inaxes][0]
            elif event.inaxes in self.axes_twin:
                vline_mover = self.vline_movers[self.axes_twin == event.inaxes][0]
            else:
                raise RuntimeError("no valid axis")

            labels_idx = int(vline_mover.x[0])
            labels = self.labels_list[labels_idx]
            labels_uniq = sorted(set(labels))
            n_clusters = len(labels_uniq)
            n_clusters_dir = pjoin(self.clustering_dir, '{}clusters'.format(n_clusters))
            if not os.path.exists(n_clusters_dir):
                os.makedirs(n_clusters_dir)

            # create soft link to corresponding labels' file
            labels_file = pjoin(self.clustering_result_dir, '{}group_labels'.format(n_clusters))
            os.system('cd {} && ln -s {} group_labels'.format(n_clusters_dir, labels_file))

            # show heatmap for rearranged data
            data_rearranged = np.zeros((0, self.data.shape[1]))
            for label in labels_uniq:
                # get subgroup data
                subgroup_data = self.data[labels == label]
                data_rearranged = np.r_[data_rearranged, subgroup_data]
            imshow(data_rearranged, 'vertices', 'subjects', 'jet', 'activation')

            plt.show()

        elif event.button == 1:
            # do something on left click
            # Here is earlier than VlineMover's left button response. So,
            # we acquire x index by event.xdata.
            if event.inaxes in self.axes or event.inaxes in self.axes_twin:
                n_labels = len(self.labels_list)
                if n_labels > 2:
                    # prepare
                    x = np.arange(n_labels)
                    x_labels = np.array([len(set(labels)) for labels in self.labels_list])
                    xticks = x[[0, -1]]
                    xticklabels = x_labels[[0, -1]]

                    # update or not
                    new_idx = int(round(event.xdata))
                    if new_idx not in xticks:
                        xticks = np.append(xticks, new_idx)
                        n_clusters = x_labels[new_idx]
                        xticklabels = np.append(xticklabels, n_clusters)

                    # plot
                    event.inaxes.set_xticks(xticks)
                    event.inaxes.set_xticklabels(xticklabels)
