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


class ClusteringVlineMoverPlotter(VlineMoverPlotter):

    def __init__(self, data, labels_list, subject_ids, subproject_dir, rFFA_vertices,
                 clustering_thr, clustering_bin, clustering_regress_mean,
                 nrows=1, ncols=1, sharex=False, sharey=False,
                 squeese=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
        super(ClusteringVlineMoverPlotter, self).__init__(nrows, ncols, sharex, sharey,
                                                          squeese, subplot_kw, gridspec_kw, **fig_kw)
        self.data = data
        self.labels_list = labels_list
        self.subject_ids = subject_ids
        self.subproject_dir = subproject_dir
        self.rFFA_vertices = rFFA_vertices

        self.clustering_thr = clustering_thr
        self.clustering_bin = clustering_bin
        self.clustering_regress_mean = clustering_regress_mean

    def _on_clicked(self, event):
        if event.button == 3:
            # do something on right click
            vline_mover = self.vline_movers[self.axes == event.inaxes][0]
            labels_idx = int(vline_mover.x[0]) - 1
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

            # show heatmap for rearranged rFFA_data
            rFFA_data_rearranged = data_rearranged[:, self.rFFA_vertices]
            if self.clustering_thr is not None:
                if self.clustering_bin:
                    rFFA_data_rearranged = (rFFA_data_rearranged > clustering_thr).astype(np.int8)
                else:
                    rFFA_data_rearranged[rFFA_data_rearranged <= clustering_thr] = clustering_thr
            if self.clustering_regress_mean and not self.clustering_bin:
                rFFA_mean_rearranged = np.atleast_2d(np.mean(rFFA_data_rearranged, 1)).T
                rFFA_data_rearranged = rFFA_data_rearranged - rFFA_mean_rearranged

            # fig, ax = plt.subplots()
            # sns.heatmap(rFFA_data_rearranged, ax=ax)
            # ax.set_xlabel('vertices')
            # ax.set_ylabel('subjects')
            # plt.savefig(pjoin(n_clusters_dir, 'rFFA_patterns_rearranged.png'))
            # plt.show()


if __name__ == '__main__':
    import os
    import nibabel as nib
    import seaborn as sns

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader
    from commontool.algorithm.tool import calc_overlap

    # predefine some variates
    # -----------------------
    # predefine parameters
    max_num = 1080
    method = 'ward'
    clustering_thr = None  # a threshold used to cut rFFA_data before clustering (default: None)
    clustering_bin = False  # If true, binarize rFFA_data according to clustering_thr
    clustering_regress_mean = True  # If true, regress mean value from rFFA_data
    subproject_name = '2mm_ward_regress'

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
    # fig, ax = plt.subplots()
    # sns.heatmap(rFFA_patterns, ax=ax)
    # ax.set_xlabel('vertices')
    # ax.set_ylabel('subjects')
    # plt.savefig(pjoin(subproject_dir, 'rFFA_patterns.png'))

    # do clustering
    labels_list = hac_scipy(rFFA_patterns, range(1, max_num+1), method,
                            output=pjoin(subproject_dir, 'hac_dendrogram.png'))

    dices = []
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
        print('Dice calculated: {0}/{1}'.format(idx+1, max_num))

    v_plotter = ClusteringVlineMoverPlotter(maps, labels_list, subject_ids, subproject_dir, rFFA_vertices,
                                            clustering_thr, clustering_bin, clustering_regress_mean)
    x = np.arange(max_num) + 1
    y = np.mean(dices, 1)
    std = np.std(dices, 1)
    v_plotter.axes[0].plot(x, y, 'b.-')
    v_plotter.axes[0].fill_between(x, y-std, y+std)
    v_plotter.add_vline_mover(x_round=True)
    v_plotter.axes[0].set_title('assessment for #subgroups')
    v_plotter.axes[0].set_xlabel('#subgroups')
    v_plotter.axes[0].set_ylabel('dice')

    plt.show()
