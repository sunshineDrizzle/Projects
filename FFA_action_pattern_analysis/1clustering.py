import numpy as np
import matplotlib.pyplot as plt

from commontool.algorithm.plot import VlineMoverPlotter


def hac_sklearn(data, n_clusters):
    from sklearn.cluster import AgglomerativeClustering

    # do hierarchical clustering on rFFA_data by using sklearn
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    clustering.fit(data)

    return clustering.labels_


def hac_scipy(data, cluster_nums, method, metric='euclidean', output=None):
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

    # do hierarchical clustering on rFFA_data and show the dendrogram by using scipy
    Z = linkage(data, method, metric)
    # plt.figure()
    # dendrogram(Z)
    labels_list = []
    for num in cluster_nums:
        labels_list.append(fcluster(Z, num, 'maxclust'))

        print('HAC finished: {}'.format(num))

    # if output is not None:
    #     plt.savefig(output)

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
    from sklearn.cluster import KMeans

    labels_list = []
    for n_clusters in cluster_nums:
        kmeans = KMeans(n_clusters, random_state=0, n_init=n_init).fit(data)
        labels_list.append(kmeans.labels_ + 1)

        print('KMeans finished: {}'.format(n_clusters))

    return labels_list


class ClusteringVlineMoverPlotter(VlineMoverPlotter):

    def __init__(self, data, labels_list, subject_ids, subproject_dir,
                 nrows=1, ncols=1, sharex=False, sharey=False,
                 squeese=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
        super(ClusteringVlineMoverPlotter, self).__init__(nrows, ncols, sharex, sharey,
                                                          squeese, subplot_kw, gridspec_kw, **fig_kw)
        self.data = data
        self.labels_list = labels_list
        self.subject_ids = subject_ids
        self.subproject_dir = subproject_dir

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
            n_clusters = max(labels)
            n_clusters_dir = pjoin(self.subproject_dir, '{}clusters'.format(n_clusters))
            if not os.path.exists(n_clusters_dir):
                os.makedirs(n_clusters_dir)

            # output
            # --------------
            data_rearranged = np.zeros((0, self.data.shape[1]))
            for label in range(1, n_clusters+1):
                # get subgroup data
                subgroup_data = self.data[labels == label]
                data_rearranged = np.r_[data_rearranged, subgroup_data]

                # get subjects' IDs in a subgroup
                subgroup_ids = self.subject_ids[labels == label]
                subgroup_ids = '\n'.join(subgroup_ids)

                # output subjects IDs
                with open(pjoin(n_clusters_dir, 'subjects{}_id'.format(label)), 'w+') as wf:
                    wf.writelines(subgroup_ids)

            # output labels
            labels_out = ' '.join([str(label) for label in labels])
            with open(pjoin(n_clusters_dir, 'subject_labels'), 'w+') as wf:
                wf.write(labels_out)

            # show heatmap for rearranged data
            fig, ax = plt.subplots()
            im = ax.imshow(data_rearranged, cmap='jet')
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('activation', rotation=-90, va="bottom")
            ax.set_xlabel('vertices')
            ax.set_ylabel('subjects')
            # plt.savefig(pjoin(n_clusters_dir, 'rFFA_patterns_rearranged.png'))
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
                    x_labels = np.array([max(labels) for labels in self.labels_list])
                    xticks = x[[0, -1]]
                    xticklabels = x_labels[[0, -1]]

                    # update or not
                    labels_idx = int(round(event.xdata))
                    if labels_idx not in xticks:
                        xticks = np.append(xticks, labels_idx)

                        labels = self.labels_list[labels_idx]
                        n_clusters = max(labels)
                        xticklabels = np.append(xticklabels, n_clusters)

                    # plot
                    event.inaxes.set_xticks(xticks)
                    event.inaxes.set_xticklabels(xticklabels)


if __name__ == '__main__':
    import os
    import nibabel as nib
    import networkx as nx

    from os.path import join as pjoin
    from community import modularity
    from commontool.io.io import CiftiReader
    from commontool.algorithm.tool import calc_overlap
    from commontool.algorithm.graph import array2adjacent_matrix

    # predefine some variates
    # -----------------------
    print('Start: predefine some variates')

    # predefine parameters
    method = 'KM'  # 'HAC', 'KM', 'LV' or 'GN'
    weight_type = ('dissimilar', 'euclidean')
    clustering_thr = None  # a threshold used to cut rFFA_data before clustering (default: None)
    clustering_bin = False  # If true, binarize rFFA_data according to clustering_thr
    clustering_regress_mean = True  # If true, regress mean value from rFFA_data
    subproject_name = '2mm_KM_init1000_regress'

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    rFFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')
    maps_path = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')

    print('Finish: predefine some variates')
    # -----------------------

    # prepare data
    # -----------------------
    print('Start: prepare data')

    # prepare rFFA patterns
    rFFA_vertices = nib.freesurfer.read_label(rFFA_label_path)
    maps_reader = CiftiReader(maps_path)
    maps = maps_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    rFFA_patterns = maps[:, rFFA_vertices]

    if clustering_thr is not None:
        if clustering_bin:
            rFFA_patterns = (rFFA_patterns > clustering_thr).astype(np.int8)
        else:
            rFFA_patterns[rFFA_patterns <= clustering_thr] = clustering_thr
    if clustering_regress_mean and not clustering_bin:
        rFFA_pattern_means = np.atleast_2d(np.mean(rFFA_patterns, 1)).T
        rFFA_patterns = rFFA_patterns - rFFA_pattern_means

    # show rFFA_patterns
    fig, ax = plt.subplots()
    im = ax.imshow(rFFA_patterns, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('activation', rotation=-90, va="bottom")
    ax.set_xlabel('vertices')
    ax.set_ylabel('subjects')
    # plt.savefig(pjoin(subproject_dir, 'rFFA_patterns.png'))

    # prepare subject ids
    subject_ids = [name.split('_')[0] for name in maps_reader.map_names()]
    subject_ids = np.array(subject_ids)

    print('Finish: prepare data')
    # -----------------------

    # structure graph
    # -----------------------
    print('Start: structure graph')

    # create adjacent matrix
    n_subjects = rFFA_patterns.shape[0]
    edges = [(i, j) for i in range(n_subjects) for j in range(i + 1, n_subjects)]
    coo_adj_matrix = array2adjacent_matrix(rFFA_patterns, weight_type,
                                           weight_normalization=True, edges=edges)

    # show adjacent matrix image
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    adj_matrix = coo_adj_matrix.toarray()
    fig, ax = plt.subplots()
    im = ax.imshow(adj_matrix, cmap='YlGn')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('weight', rotation=-90, va="bottom")
    ax.set_xticks(np.arange(n_subjects))
    ax.set_xticklabels(subject_ids)
    ax.set_yticks(np.arange(n_subjects))
    ax.set_yticklabels(subject_ids)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
    fig.tight_layout()

    # create graph
    graph = nx.Graph()
    graph.add_nodes_from(range(n_subjects))
    graph.add_weighted_edges_from(zip(coo_adj_matrix.row, coo_adj_matrix.col, coo_adj_matrix.data))

    print('Finish: structure graph')
    # -----------------------

    # do clustering
    # -----------------------
    print('Start: do clustering')

    if method == 'LV':
        labels_list = louvain_community(graph)
    elif method == 'GN':
        labels_list = girvan_newman_community(graph, 200)
    elif method == 'HAC':
        labels_list = hac_scipy(rFFA_patterns, range(1, 201), 'ward',
                                output=pjoin(subproject_dir, 'hac_dendrogram.png'))
    elif method == 'KM':
        labels_list = k_means(rFFA_patterns, range(1, 201), 1000)
    else:
        raise RuntimeError('The method-{} is not supported!'.format(method))

    print('Finish: do clustering')
    # -----------------------

    # calculate assessment curve
    # -----------------------
    print('Start: calculate assessment curve')

    dices = []
    modularities = []
    n_labels = len(labels_list)
    for idx, labels in enumerate(labels_list):
        sub_dices = []
        for label in range(1, max(labels)+1):
            subgroup_rFFA_patterns = np.atleast_2d(maps[labels == label])[:, rFFA_vertices]
            subgroup_rFFA_patterns_mean = np.mean(subgroup_rFFA_patterns, 0)

            collection1 = np.where(subgroup_rFFA_patterns_mean > 2.3)[0]
            collection2s = [np.where(i > 2.3)[0] for i in subgroup_rFFA_patterns]
            tmp_dices = map(lambda c2: calc_overlap(collection1, c2), collection2s)
            sub_dices.extend(tmp_dices)
        dices.append(sub_dices)
        partition_dict = {k: v for k, v in enumerate(labels)}
        modularities.append(modularity(partition_dict, graph, weight='weight'))
        print('Assessment calculated: {0}/{1}'.format(idx+1, n_labels))

    # plot dice assessment curve
    v_plotter = ClusteringVlineMoverPlotter(rFFA_patterns, labels_list, subject_ids, subproject_dir)
    x = np.arange(n_labels)
    x_labels = np.array([max(labels) for labels in labels_list])
    y = np.mean(dices, 1)
    std = np.std(dices, 1)
    v_plotter.axes[0].plot(x, y, 'b.-')
    v_plotter.axes[0].fill_between(x, y-std, y+std, alpha=0.5)
    v_plotter.axes[0].set_title('assessment for #subgroups')
    v_plotter.axes[0].set_xlabel('#subgroups')
    if n_labels > 2:
        vline_idx = int(n_labels/2)
        v_plotter.axes[0].set_xticks(x[[0, vline_idx, -1]])
        v_plotter.axes[0].set_xticklabels(x_labels[[0, vline_idx, -1]])
    else:
        vline_idx = 0
        v_plotter.axes[0].set_xticks(x)
        v_plotter.axes[0].set_xticklabels(x_labels)
    v_plotter.axes[0].set_ylabel('dice', color='b')
    v_plotter.axes[0].tick_params('y', colors='b')
    plt.setp(v_plotter.axes[0].get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    # plot modularity assessment curve in a twin axis
    # https://matplotlib.org/examples/api/two_scales.html
    v_plotter.add_twinx(0)
    v_plotter.axes_twin[0].plot(x, modularities, 'r.-')
    v_plotter.axes_twin[0].set_ylabel('modularity', color='r')
    v_plotter.axes_twin[0].tick_params('y', colors='r')

    # add vline mover
    v_plotter.add_vline_mover(vline_idx=vline_idx, x_round=True)
    v_plotter.figure.tight_layout()

    print('Finish: calculate assessment curve')
    # -----------------------

    plt.show()
