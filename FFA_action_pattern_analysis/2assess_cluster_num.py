import os
import numpy as np
import matplotlib.pyplot as plt

from os.path import join as pjoin
from commontool.algorithm.plot import VlineMoverPlotter


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
            labels_file = pjoin(self.clustering_result_dir, '{}subject_labels'.format(n_clusters))
            os.system('cd {} && ln -s {} subject_labels'.format(n_clusters_dir, labels_file))

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


if __name__ == '__main__':
    import nibabel as nib

    from scipy import stats
    from commontool.io.io import CiftiReader
    from commontool.algorithm.tool import elbow_score
    from commontool.algorithm.plot import imshow

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'both'  # 'lh', 'rh', 'both'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    weight_type = ('dissimilar', 'euclidean')
    clustering_method = 'HAC_ward_euclidean'
    max_cluster_num = 50
    # dice, modularity, silhouette, gap statistic, elbow_inner_standard
    # elbow_inner_centroid, elbow_inner_pairwise, elbow_inter_centroid, elbow_inter_pairwise
    assessment_metric_pairs = [
        # ['dice', 'modularity'],
        ['elbow_inner_standard'],
        # ['modularity'],
        # ['silhouette'],
        # ['gap statistic']
    ]
    assessments_dict = dict()
    for metric_pair in assessment_metric_pairs:
        for metric in metric_pair:
            assessments_dict[metric] = []
    is_graph_needed = True if 'modularity' in assessments_dict.keys() else False

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    analysis_dir = pjoin(project_dir, 's2_25_zscore')
    clustering_dir = pjoin(analysis_dir, clustering_method)
    clustering_result_dir = pjoin(clustering_dir, 'clustering_results')
    FFA_label_files = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label')
    maps_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    FFA_pattern_files = pjoin(analysis_dir, '{}FFA_patterns.nii.gz')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare FFA patterns
    reader = CiftiReader(maps_file)
    if hemi == 'both':
        lFFA_vertices = nib.freesurfer.read_label(FFA_label_files.format('l'))
        rFFA_vertices = nib.freesurfer.read_label(FFA_label_files.format('r'))
        lFFA_maps = reader.get_data(brain_structure['lh'], True)[:, lFFA_vertices]
        rFFA_maps = reader.get_data(brain_structure['rh'], True)[:, rFFA_vertices]
        FFA_maps = np.c_[lFFA_maps, rFFA_maps]
        lFFA_patterns = nib.load(FFA_pattern_files.format('l')).get_data()
        rFFA_patterns = nib.load(FFA_pattern_files.format('r')).get_data()
        FFA_patterns = np.c_[lFFA_patterns, rFFA_patterns]
    else:
        FFA_vertices = nib.freesurfer.read_label(FFA_label_files.format(hemi[0]))
        FFA_maps = reader.get_data(brain_structure[hemi], True)[:, FFA_vertices]
        FFA_patterns = nib.load(FFA_pattern_files.format(hemi[0])).get_data()
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

    print('Start: calculate assessments')
    # -----------------------
    labels_list = []
    labels_files = [pjoin(clustering_result_dir, item) for item in os.listdir(clustering_result_dir)
                    if 'subject_labels' in item]
    for labels_file in labels_files:
        labels_list.append(np.array(open(labels_file).read().split(' '), dtype=np.uint16))
    labels_list.sort(key=lambda x: len(set(x)))
    cluster_nums = np.array([len(set(labels)) for labels in labels_list])
    indices = np.where(cluster_nums > max_cluster_num)[0]
    if indices:
        end_idx = indices[0]
        labels_list = labels_list[:end_idx]
        cluster_nums = cluster_nums[:end_idx]
    n_labels = len(labels_list)
    for idx, labels in enumerate(labels_list):
        labels_uniq = np.unique(labels)
        for metric in assessments_dict.keys():
            if metric == 'dice':
                from commontool.algorithm.tool import calc_overlap

                sub_dices = []
                for label in labels_uniq:
                    subgroup_FFA_maps = np.atleast_2d(FFA_maps[labels == label])
                    subgroup_FFA_maps_mean = np.mean(subgroup_FFA_maps, 0)

                    collection1 = np.where(subgroup_FFA_maps_mean > 2.3)[0]
                    collection2s = [np.where(i > 2.3)[0] for i in subgroup_FFA_maps]
                    tmp_dices = map(lambda c2: calc_overlap(collection1, c2), collection2s)
                    sub_dices.extend(tmp_dices)
                assessments_dict[metric].append(sub_dices)

            elif metric == 'modularity':
                from community import modularity

                partition_dict = {k: v for k, v in enumerate(labels)}
                assessments_dict[metric].append(modularity(partition_dict, graph, weight='weight'))

            elif metric == 'silhouette':
                from sklearn.metrics import silhouette_score

                # https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
                # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
                if len(labels_uniq) > 1:
                    assessments_dict[metric].append(silhouette_score(FFA_patterns, labels,
                                                                     metric=weight_type[1], random_state=0))

            elif 'elbow' in metric:
                tmp = metric.split('_')
                assessment = elbow_score(FFA_patterns, labels, metric=weight_type[1],
                                         type=(tmp[1], tmp[2]))
                assessments_dict[metric].append(assessment)

            elif metric == 'gap statistic':
                pass

            else:
                raise RuntimeError("{} isn't a valid assessment metric".format(metric))

        print('Assessment calculated: {0}/{1}'.format(idx + 1, n_labels))

    if 'gap statistic' in assessments_dict.keys():
        from FFA_action_pattern_analysis.clustering1 import hac_scipy, k_means

        if 'HAC' in clustering_method:
            cluster_method = hac_scipy
        elif 'KM' in clustering_method:
            cluster_method = k_means
        else:
            raise RuntimeError("analysis-{} isn't supported at present!".format(clustering_method))

        labels_list, gaps, s, k_selected = gap_stat_mine(FFA_patterns, cluster_nums,
                                                         cluster_method=cluster_method)
        assessments_dict['gap statistic'] = (gaps, s, k_selected)

    x = np.arange(n_labels)
    x_labels = cluster_nums
    vline_plotter_holder = []
    for metric_pair in assessment_metric_pairs:
        # plot assessment curve
        v_plotter = ClusteringVlineMoverPlotter(FFA_patterns, labels_list, clustering_dir, clustering_result_dir)

        if metric_pair[0] == 'dice':
            y = np.mean(assessments_dict[metric_pair[0]], 1)
            sem = stats.sem(assessments_dict[metric_pair[0]], 1)
            v_plotter.axes[0].plot(x, y, 'b.-')
            v_plotter.axes[0].fill_between(x, y-sem, y+sem, alpha=0.5)

        elif 'elbow' in metric_pair[0]:
            y = assessments_dict[metric_pair[0]]
            v_plotter.axes[0].plot(x, y, 'b.-')

            x1 = x[:-1]
            y1 = [y[i] - y[i + 1] for i in x1]
            fig1, ax1 = plt.subplots()
            ax1.plot(x1, y1, 'b.-')
            ax1.set_title('assessment for #subgroups')
            ax1.set_xlabel('#subgroups')
            ax1.set_ylabel(metric_pair[0] + "'")
            if len(x1) > 20:
                middle_idx = int(len(x1) / 2)
                ax1.set_xticks(x1[[0, middle_idx, -1]])
                ax1.set_xticklabels(x_labels[1:][[0, middle_idx, -1]])
            else:
                ax1.set_xticks(x1)
                ax1.set_xticklabels(x_labels[1:])

            x2 = x1[:-1]
            y2 = [y1[i] - y1[i + 1] for i in x2]
            fig2, ax2 = plt.subplots()
            ax2.plot(x2, y2, 'b.-')
            ax2.set_title('assessment for #subgroups')
            ax2.set_xlabel('#subgroups')
            ax2.set_ylabel(metric_pair[0] + "''")
            if len(x2) > 20:
                middle_idx = int(len(x2) / 2)
                ax2.set_xticks(x2[[0, middle_idx, -1]])
                ax2.set_xticklabels(x_labels[1:-1][[0, middle_idx, -1]])
            else:
                ax2.set_xticks(x2)
                ax2.set_xticklabels(x_labels[1:-1])

        elif metric_pair[0] == 'gap statistic':
            v_plotter.axes[0].plot(x, assessments_dict[metric_pair[0]][0], 'b.-')
            v_plotter.axes[0].fill_between(x, assessments_dict[metric_pair[0]][0] - assessments_dict[metric_pair[0]][1],
                                           assessments_dict[metric_pair[0]][0] + assessments_dict[metric_pair[0]][1],
                                           alpha=0.5)
        elif metric_pair[0] == 'silhouette':
            if x_labels[0] == 1:
                v_plotter.axes[0].plot(x[1:], assessments_dict[metric_pair[0]], 'b.-')
            else:
                v_plotter.axes[0].plot(x, assessments_dict[metric_pair[0]], 'b.-')
        else:
            v_plotter.axes[0].plot(x, assessments_dict[metric_pair[0]], 'b.-')

        v_plotter.axes[0].set_title('assessment for #subgroups')
        v_plotter.axes[0].set_xlabel('#subgroups')
        if n_labels > 2:
            if metric_pair[0] == 'gap statistic':
                vline_idx = np.where(x_labels == assessments_dict[metric_pair[0]][2])[0][0]
            else:
                vline_idx = int(n_labels / 2)
            v_plotter.axes[0].set_xticks(x[[0, vline_idx, -1]])
            v_plotter.axes[0].set_xticklabels(x_labels[[0, vline_idx, -1]])
            plt.setp(v_plotter.axes[0].get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
        else:
            if metric_pair[0] == 'gap statistic':
                vline_idx = np.where(x_labels == assessments_dict[metric_pair[0]][2])[0][0]
            else:
                vline_idx = 0
            v_plotter.axes[0].set_xticks(x)
            v_plotter.axes[0].set_xticklabels(x_labels)
        v_plotter.axes[0].set_ylabel(metric_pair[0], color='b')
        v_plotter.axes[0].tick_params('y', colors='b')

        if len(metric_pair) == 2:
            # plot another assessment curve in a twin axis
            # https://matplotlib.org/examples/api/two_scales.html
            v_plotter.add_twinx(0)
            if metric_pair[1] == 'dice':
                y = np.mean(assessments_dict[metric_pair[1]], 1)
                sem = stats.sem(assessments_dict[metric_pair[1]], 1)
                v_plotter.axes_twin[0].plot(x, y, 'r.-')
                v_plotter.axes_twin[0].fill_between(x, y - sem, y + sem, alpha=0.5)
            elif metric_pair[1] == 'gap statistic':
                raise RuntimeError("{} can't be plot at twin axis at present!".format(metric_pair[1]))
            elif metric_pair[1] == 'silhouette':
                if x_labels[0] == 1:
                    v_plotter.axes_twin[0].plot(x[1:], assessments_dict[metric_pair[1]], 'b.-')
                else:
                    v_plotter.axes_twin[0].plot(x, assessments_dict[metric_pair[1]], 'b.-')
            else:
                v_plotter.axes_twin[0].plot(x, assessments_dict[metric_pair[1]], 'r.-')

            v_plotter.axes_twin[0].set_ylabel(metric_pair[1], color='r')
            v_plotter.axes_twin[0].tick_params('y', colors='r')

        v_plotter.add_vline_mover(vline_idx=vline_idx, x_round=True)
        v_plotter.figure.tight_layout()
        vline_plotter_holder.append(v_plotter)
    # -----------------------
    print('Finish: calculate assessments')

    plt.show()
