import numpy as np
import matplotlib.pyplot as plt


def get_roi_pattern():
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from FFA_pattern.tool import get_roi_pattern

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'rh'
    zscore = True  # If true, do z-score on each subject's FFA pattern
    thr = None  # a threshold used to cut FFA_data before clustering (default: None)
    bin = False  # If true, binarize FFA_data according to clustering_thr
    size_min = 0  # only work with threshold

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    trg_dir = pjoin(proj_dir, f'analysis/clustering_{hemi}')
    roi_file = pjoin(proj_dir, f'data/HCP/label/MMPprob_OFA_FFA_thr1_{hemi}.label')
    activ_file = pjoin(trg_dir, f'activation_{hemi}.nii.gz')
    # geo_files = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
    #             'HCP_S1200_GroupAvg_v1/S1200.{}.white_MSMAll.32k_fs_LR.surf.gii'
    geo_files = None
    # mask_file = pjoin(project_dir, f'data/HCP_face-avg/s2/patches_15/crg2.3/{hemi[0]}FFA_patch_maps_lt15.nii.gz')
    mask_file = None
    # -----------------------
    print('Finish: predefine some variates')

    if mask_file is not None:
        mask = nib.load(mask_file).get_data().squeeze().T != 0
    else:
        mask = None
    if geo_files is None:
        faces = None
    else:
        raise NotImplementedError

    activ = nib.load(activ_file).get_data().squeeze().T
    roi = nib.freesurfer.read_label(roi_file)
    roi_patterns = get_roi_pattern(activ, roi, zscore, thr, bin, size_min, faces, mask)

    np.save(pjoin(trg_dir, f'roi_pattern_{hemi}.npy'), roi_patterns)


def hac_sklearn(data, n_clusters):
    from sklearn.cluster import AgglomerativeClustering

    # do hierarchical clustering on FFA_data by using sklearn
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(data)

    return clustering.labels_


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


def clustering():
    import os
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.algorithm.plot import imshow
    from commontool.algorithm.cluster import hac_scipy

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'rh'  # 'lh', 'rh', 'both'
    clustering_method = 'HAC_ward_euclidean'  # 'HAC_average_dice', 'KM', 'LV', 'GN'
    is_graph_needed = True if clustering_method in ('LV', 'GN') else False
    weight_type = ('dissimilar', 'euclidean')  # only work when is_graph_needed is True

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    analysis_dir = pjoin(project_dir, 'analysis/2clustering/zscore_FFA_25')
    clustering_result_dir = pjoin(analysis_dir, f'{clustering_method}_{hemi[0]}FFA/clustering_results')
    if not os.path.exists(clustering_result_dir):
        os.makedirs(clustering_result_dir)
    FFA_label_files = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label')
    FFA_pattern_files = pjoin(analysis_dir, '{}FFA_patterns.nii.gz')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare FFA patterns
    if hemi == 'both':
        lFFA_patterns = nib.load(FFA_pattern_files.format('l')).get_data()
        rFFA_patterns = nib.load(FFA_pattern_files.format('r')).get_data()
        FFA_patterns = np.c_[lFFA_patterns, rFFA_patterns]
    else:
        FFA_patterns = nib.load(FFA_pattern_files.format(hemi[0])).get_data()

    # show FFA_patterns
    imshow(FFA_patterns, 'vertices', 'subjects', 'jet', 'pattern')
    # -----------------------
    print('Finish: prepare data')

    # structure graph
    # -----------------------
    if is_graph_needed:
        from commontool.algorithm.graph import array2graph

        print('Start: structure graph')
        graph = array2graph(FFA_patterns, weight_type, edges='upper_right_triangle')
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
                                out_path=pjoin(clustering_result_dir, 'hac_dendrogram.png'))
    elif clustering_method == 'KM':
        labels_list = k_means(FFA_patterns, range(1, 51), 10)
    else:
        raise RuntimeError('The clustering_method-{} is not supported!'.format(clustering_method))

    # output labels
    for labels in labels_list:
        n_label = len(set(labels))
        labels_out = ' '.join([str(label) for label in labels])
        with open(pjoin(clustering_result_dir, '{}group_labels'.format(n_label)), 'w+') as wf:
            wf.write(labels_out)
    # -----------------------
    print('Finish: do clustering')

    plt.show()


if __name__ == '__main__':
    get_roi_pattern()
