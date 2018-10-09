import numpy as np
import matplotlib.pyplot as plt

from commontool.algorithm.plot import VlineMoverPlotter


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
            ax.imshow(data_rearranged)
            ax.set_xlabel('vertices')
            ax.set_ylabel('subjects')
            # plt.savefig(pjoin(n_clusters_dir, 'rFFA_patterns_rearranged.png'))
            plt.show()


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
    # predefine parameters
    detection_method = 'GN'
    weight_type = ('dissimilar', 'euclidean')
    clustering_thr = None  # a threshold used to cut rFFA_data before clustering (default: None)
    clustering_bin = False  # If true, binarize rFFA_data according to clustering_thr
    clustering_regress_mean = True  # If true, regress mean value from rFFA_data
    subproject_name = '2mm_{0}_{1}_regress'.format(detection_method, weight_type[1])

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    rFFA_label_path = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')
    maps_path = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    # -----------------------

    # get data
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

    subject_ids = [name.split('_')[0] for name in maps_reader.map_names()]
    subject_ids = np.array(subject_ids)

    # plot heatmap for rFFA_patterns
    fig, ax = plt.subplots()
    ax.imshow(rFFA_patterns)
    ax.set_xlabel('vertices')
    ax.set_ylabel('subjects')
    # plt.savefig(pjoin(subproject_dir, 'rFFA_patterns.png'))

    # create adjacent matrix
    n_subjects = rFFA_patterns.shape[0]
    edges = [(i, j) for i in range(n_subjects) for j in range(i+1, n_subjects)]
    coo_adj_matrix = array2adjacent_matrix(rFFA_patterns, weight_type, weight_normalization=True, edges=edges)

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # community detection
    graph = nx.Graph()
    graph.add_nodes_from(range(n_subjects))
    graph.add_weighted_edges_from(zip(coo_adj_matrix.row, coo_adj_matrix.col, coo_adj_matrix.data))
    if detection_method == 'louvain':
        labels_list = louvain_community(graph)
    elif detection_method == 'GN':
        labels_list = girvan_newman_community(graph, 1080)
    else:
        raise RuntimeError('The detection method-{} is not supported!'.format(detection_method))

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
        print('Dice calculated: {0}/{1}'.format(idx+1, n_labels))

    v_plotter = ClusteringVlineMoverPlotter(rFFA_patterns, labels_list, subject_ids, subproject_dir)
    x = np.arange(n_labels)
    x_labels = [max(labels) for labels in labels_list]
    y = np.mean(dices, 1)
    std = np.std(dices, 1)
    v_plotter.axes[0].plot(x, y, 'b.-')
    v_plotter.axes[0].fill_between(x, y-std, y+std)
    v_plotter.axes[0].set_title('assessment for #subgroups')
    v_plotter.axes[0].set_xlabel('#subgroups')
    v_plotter.axes[0].set_xticks(x)
    v_plotter.axes[0].set_xticklabels(x_labels)
    v_plotter.axes[0].set_ylabel('dice', color='b')
    v_plotter.axes[0].tick_params('y', colors='b')

    # https://matplotlib.org/examples/api/two_scales.html
    v_plotter.add_twinx(0)
    v_plotter.axes_twin[0].plot(x, modularities, 'r.-')
    v_plotter.axes_twin[0].set_ylabel('modularity', color='r')
    v_plotter.axes_twin[0].tick_params('y', colors='r')
    v_plotter.add_vline_mover(x_round=True)
    v_plotter.figure.tight_layout()

    plt.show()
