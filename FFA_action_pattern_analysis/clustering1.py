import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def hac_sklearn(data, n_clusters):
    from sklearn.cluster import AgglomerativeClustering

    # do hierarchical clustering on FFA_data by using sklearn
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(data)

    return clustering.labels_


def hac_scipy(data, cluster_nums, method='ward', metric='euclidean', output=None):
    """

    :param data:
    :param cluster_nums: sequence | iterator
        Each element is the number of clusters that HAC generate.
    :param method:
    :param metric:
    :param output:

    :return: labels_list: list
        label results of each cluster_num
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

    # do hierarchical clustering on FFA_data and show the dendrogram by using scipy
    Z = linkage(data, method, metric)
    labels_list = []
    for num in cluster_nums:
        labels_list.append(fcluster(Z, num, 'maxclust'))
        print('HAC finished: {}'.format(num))

    if output is not None:
        plt.figure()
        dendrogram(Z)
        plt.savefig(output)

    return labels_list


def verify_same_effect_between_two_tools(data):
    labels_sklearn = hac_sklearn(data, 2)
    labels_scipy = hac_scipy(data, 2, 'ward')
    # the label number 0 in labels_sklearn is corresponding to 2 in labels_scipy
    # the label number 1 in labels_sklearn is corresponding to 1 in labels_sicpy
    sklearn_1s = np.where(labels_sklearn == 1)[0]
    scipy_1s = np.where(labels_scipy == 1)[0]
    diff = sklearn_1s - scipy_1s
    print(max(diff), min(diff))


def louvain_community(graph):
    """
    https://python-louvain.readthedocs.io/en/latest/
    https://en.wikipedia.org/wiki/Louvain_Modularity

    :param graph:
    :return:
    """
    from community import best_partition

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
    import itertools
    from networkx import edge_betweenness_centrality as betweenness
    from networkx.algorithms.community.centrality import girvan_newman

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
    from sklearn.cluster import KMeans

    labels_list = []
    for n_clusters in cluster_nums:
        kmeans = KMeans(n_clusters, random_state=0, n_init=n_init).fit(data)
        labels_list.append(kmeans.labels_ + 1)
        print('KMeans finished: {}'.format(n_clusters))

    return labels_list


def map2pattern(maps, clustering_thr=None, clustering_bin=False, clustering_zscore=False):
    patterns = maps.copy()
    if clustering_thr is not None:
        if clustering_bin:
            patterns = (patterns > clustering_thr).astype(np.int8)
        else:
            patterns[patterns <= clustering_thr] = clustering_thr
    if clustering_zscore and not clustering_bin:
        patterns = stats.zscore(patterns, 1)

    return patterns


if __name__ == '__main__':
    import os
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader
    from commontool.algorithm.plot import imshow

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    clustering_method = 'HAC_single_dice'  # 'HAC_average_dice', 'KM', 'LV', 'GN'
    clustering_thr = 2.3  # a threshold used to cut FFA_data before clustering (default: None)
    clustering_bin = True  # If true, binarize FFA_data according to clustering_thr
    clustering_zscore = False  # If true, do z-score on each subject's FFA pattern
    # brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    is_graph_needed = True if clustering_method in ('LV', 'GN') else False

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    clustering_dir = pjoin(project_dir, '2mm_{}_thr2.3_bin/clustering_results'.format(clustering_method))
    if not os.path.exists(clustering_dir):
        os.makedirs(clustering_dir)
    # FFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm.label')
    lFFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm.label')
    rFFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')
    maps_path = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare FFA patterns
    # FFA_vertices = nib.freesurfer.read_label(FFA_label_path)
    lFFA_vertices = nib.freesurfer.read_label(lFFA_label_path)
    rFFA_vertices = nib.freesurfer.read_label(rFFA_label_path)
    maps_reader = CiftiReader(maps_path)
    # maps = maps_reader.get_data(brain_structure, True)
    lmaps = maps_reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
    rmaps = maps_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    # FFA_maps = maps[:, FFA_vertices]
    lFFA_maps = lmaps[:, lFFA_vertices]
    rFFA_maps = rmaps[:, rFFA_vertices]
    FFA_maps = np.c_[lFFA_maps, rFFA_maps]

    # FFA_patterns = map2pattern(FFA_maps, clustering_thr, clustering_bin, clustering_zscore)
    lFFA_patterns = map2pattern(lFFA_maps, clustering_thr, clustering_bin, clustering_zscore)
    rFFA_patterns = map2pattern(rFFA_maps, clustering_thr, clustering_bin, clustering_zscore)
    FFA_patterns = np.c_[lFFA_patterns, rFFA_patterns]

    # show FFA_patterns
    imshow(FFA_patterns, 'vertices', 'subjects', 'jet', 'activation')
    # -----------------------
    print('Finish: prepare data')

    # structure graph
    # -----------------------
    if is_graph_needed:
        from commontool.algorithm.graph import array2graph

        print('Start: structure graph')
        graph = array2graph(FFA_patterns, ('dissimilar', 'euclidean'), edges='upper_right_triangle')
        print('Finish: structure graph')
    else:
        graph = None
    # -----------------------

    print('Start: do clustering')
    # -----------------------
    if clustering_method == 'LV':
        labels_list = louvain_community(graph)
    elif clustering_method == 'GN':
        labels_list = girvan_newman_community(graph, 50)
    elif 'HAC' in clustering_method:
        values = clustering_method.split('_')
        labels_list = hac_scipy(FFA_patterns, range(1, 51), method=values[1], metric=values[2],
                                output=pjoin(clustering_dir, 'hac_dendrogram.png'))
    elif clustering_method == 'KM':
        labels_list = k_means(FFA_patterns, range(1, 51), 10)
    else:
        raise RuntimeError('The clustering_method-{} is not supported!'.format(clustering_method))

    # output labels
    for labels in labels_list:
        n_label = len(set(labels))
        labels_out = ' '.join([str(label) for label in labels])
        with open(pjoin(clustering_dir, '{}subject_labels'.format(n_label)), 'w+') as wf:
            wf.write(labels_out)
    # -----------------------
    print('Finish: do clustering')

    plt.show()
