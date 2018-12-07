import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from commontool.algorithm.triangular_mesh import get_n_ring_neighbor
from FFA_action_pattern_analysis.tmp_tool.patch.get_patches import get_patch_by_crg


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


def get_roi_patterns(maps, roi, zscore=False, thr=None, bin=False, size_min=0, faces=None, mask=None):
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


if __name__ == '__main__':
    import os
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, GiftiReader
    from commontool.algorithm.plot import imshow

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'both'  # 'lh', 'rh', 'both'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    clustering_method = 'HAC_ward_euclidean'  # 'HAC_average_dice', 'KM', 'LV', 'GN'
    zscore = True  # If true, do z-score on each subject's FFA pattern
    thr = None  # a threshold used to cut FFA_data before clustering (default: None)
    bin = False  # If true, binarize FFA_data according to clustering_thr
    size_min = 0  # only work with threshold
    is_graph_needed = True if clustering_method in ('LV', 'GN') else False

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    clustering_dir = pjoin(project_dir,
                           '2mm_15_{}_zscore/clustering_results'.format(clustering_method))
    if not os.path.exists(clustering_dir):
        os.makedirs(clustering_dir)
    FFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/{}FFA_2mm_15.label')
    maps_path = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    lh_geo_file = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
                  'HCP_S1200_GroupAvg_v1/S1200.L.white_MSMAll.32k_fs_LR.surf.gii'
    rh_geo_file = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
                  'HCP_S1200_GroupAvg_v1/S1200.R.white_MSMAll.32k_fs_LR.surf.gii'
    # mask_file = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/crg2.3/{}FFA_patch_maps_lt15.nii.gz')
    mask_file = None
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare FFA patterns
    reader = CiftiReader(maps_path)
    if hemi == 'both':
        if mask_file is not None:
            lh_mask = nib.load(mask_file.format('l')).get_data() != 0
            rh_mask = nib.load(mask_file.format('r')).get_data() != 0
        else:
            lh_mask = None
            rh_mask = None
        lh_geo_reader = GiftiReader(lh_geo_file)
        rh_geo_reader = GiftiReader(rh_geo_file)
        lFFA_vertices = nib.freesurfer.read_label(FFA_label_path.format('l'))
        rFFA_vertices = nib.freesurfer.read_label(FFA_label_path.format('r'))
        lFFA_patterns = get_roi_patterns(reader.get_data(brain_structure['lh'], True), lFFA_vertices,
                                         zscore, thr, bin, size_min, lh_geo_reader.faces, lh_mask)
        rFFA_patterns = get_roi_patterns(reader.get_data(brain_structure['rh'], True), rFFA_vertices,
                                         zscore, thr, bin, size_min, rh_geo_reader.faces, rh_mask)
        FFA_patterns = np.c_[lFFA_patterns, rFFA_patterns]
    else:
        if hemi == 'lh':
            geo_reader = GiftiReader(lh_geo_file)
        elif hemi == 'rh':
            geo_reader = GiftiReader(rh_geo_file)
        else:
            raise RuntimeError("invalid hemi: {}".format(hemi))

        if mask_file is not None:
            mask = nib.load(mask_file.format(hemi[0])).get_data() != 0
        else:
            mask = None

        FFA_vertices = nib.freesurfer.read_label(FFA_label_path.format(hemi[0]))
        FFA_patterns = get_roi_patterns(reader.get_data(brain_structure[hemi], True), FFA_vertices,
                                        zscore, thr, bin, size_min, geo_reader.faces, mask)

    # show FFA_patterns
    # imshow(FFA_patterns, 'vertices', 'subjects', 'jet', 'activation')
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
