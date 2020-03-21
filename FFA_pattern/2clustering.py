def get_roi_pattern():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from FFA_pattern.tool import get_roi_pattern
    from commontool.io.io import CiftiReader

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'lh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    zscore = True  # If true, do z-score on each subject's ROI pattern
    thr = None  # a threshold used to cut FFA_data before clustering (default: None)
    bin = False  # If true, binarize FFA_data according to clustering_thr
    size_min = 0  # only work with threshold

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    trg_dir = pjoin(proj_dir, f'analysis/s2/lh/zscore')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)
    roi_file = pjoin(proj_dir, f'data/HCP/label/MMPprob_OFA_FFA_thr1_{hemi}.label')
    activ_file = pjoin(proj_dir, f'analysis/s2/activation.dscalar.nii')
    # geo_files = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
    #             'HCP_S1200_GroupAvg_v1/S1200.{}.white_MSMAll.32k_fs_LR.surf.gii'
    geo_files = None
    # mask_file = pjoin(proj_dir, f'data/HCP_face-avg/s2/patches_15/crg2.3/{hemi[0]}FFA_patch_maps_lt15.nii.gz')
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

    # activ = nib.load(activ_file).get_data().squeeze().T
    activ = CiftiReader(activ_file).get_data(brain_structure[hemi], True)
    roi = nib.freesurfer.read_label(roi_file)
    roi_patterns = get_roi_pattern(activ, roi, zscore, thr, bin, size_min, faces, mask)

    np.save(pjoin(trg_dir, f'roi_patterns.npy'), roi_patterns)


def clustering():
    import os
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import imshow
    from commontool.algorithm.cluster import hac_scipy
    from FFA_pattern.tool import k_means, louvain_community, girvan_newman_community

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    clustering_method = 'HAC_ward_euclidean'  # 'HAC_average_dice', 'KM', 'LV', 'GN', 'HAC_average_correlation', 'HAC_ward_euclidean'
    max_n_cluster = 100
    is_graph_needed = True if clustering_method in ('LV', 'GN') else False
    weight_type = ('dissimilar', 'euclidean')  # only work when is_graph_needed is True

    # predefine paths
    src_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/s2/lh/zscore'
    trg_dir = pjoin(src_dir, f'{clustering_method}/results')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    roi_pattern_file = pjoin(src_dir, 'roi_patterns.npy')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare patterns
    roi_patterns = np.load(roi_pattern_file)

    # show patterns
    imshow(roi_patterns, 'vertices', 'subjects', 'jet', 'pattern')
    # -----------------------
    print('Finish: prepare data')

    # structure graph
    # -----------------------
    if is_graph_needed:
        from commontool.algorithm.graph import array2graph

        print('Start: structure graph')
        graph = array2graph(roi_patterns, weight_type, edges='upper_right_triangle')
        print('Finish: structure graph')
    else:
        graph = None
    # -----------------------

    print('Start: do clustering')
    # -----------------------
    if clustering_method == 'LV':
        labels_list = louvain_community(graph)
    elif clustering_method == 'GN':
        labels_list = girvan_newman_community(graph, max_n_cluster)
    elif 'HAC' in clustering_method:
        values = clustering_method.split('_')
        labels_list = hac_scipy(roi_patterns, range(1, max_n_cluster+1), method=values[1], metric=values[2],
                                out_path=pjoin(trg_dir, 'hac_dendrogram.png'))
    elif clustering_method == 'KM':
        labels_list = k_means(roi_patterns, range(1, max_n_cluster+1), 10)
    else:
        raise RuntimeError('The clustering_method-{} is not supported!'.format(clustering_method))

    # output labels
    for labels in labels_list:
        n_label = len(set(labels))
        labels_out = ' '.join([str(label) for label in labels])
        with open(pjoin(trg_dir, '{}group_labels'.format(n_label)), 'w') as wf:
            wf.write(labels_out)
    # -----------------------
    print('Finish: do clustering')

    plt.show()


def assess_n_cluster():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy import stats
    from matplotlib import pyplot as plt
    from commontool.io.io import CiftiReader
    from commontool.algorithm.tool import elbow_score
    from FFA_pattern.tool import ClusteringVlineMoverPlotter

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'lh'  # 'lh', 'rh', 'both'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    weight_type = ('dissimilar', 'euclidean')
    clustering_method = 'HAC_ward_euclidean'
    max_n_cluster = 100
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
    is_graph_needed = 'modularity' in assessments_dict.keys()

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    pattern_dir = pjoin(proj_dir, 'analysis/s2/lh/zscore')
    meth_dir = pjoin(pattern_dir, clustering_method)
    result_dir = pjoin(meth_dir, 'results')
    roi_files = pjoin(proj_dir, 'data/HCP/label/MMPprob_OFA_FFA_thr1_{}.label')
    activ_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    pattern_file = pjoin(pattern_dir, 'roi_patterns.npy')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare patterns
    reader = CiftiReader(activ_file)
    if hemi == 'both':
        roi_lh = nib.freesurfer.read_label(roi_files.format('lh'))
        roi_rh = nib.freesurfer.read_label(roi_files.format('rh'))
        activ_roi_lh = reader.get_data(brain_structure['lh'], True)[:, roi_lh]
        activ_roi_rh = reader.get_data(brain_structure['rh'], True)[:, roi_rh]
        # activ_roi_lh = nib.load(activ_file.format('lh')).get_data().squeeze().T[:, roi_lh]
        # activ_roi_rh = nib.load(activ_file.format('rh')).get_data().squeeze().T[:, roi_rh]
        activ_roi = np.c_[activ_roi_lh, activ_roi_rh]
    else:
        roi = nib.freesurfer.read_label(roi_files.format(hemi))
        activ_roi = reader.get_data(brain_structure[hemi], True)[:, roi]
        # activ_roi = nib.load(activ_file).get_data().squeeze().T[:, roi]
    patterns = np.load(pattern_file)
    # -----------------------
    print('Finish: prepare data')

    # structure graph
    # -----------------------
    if is_graph_needed:
        from commontool.algorithm.graph import array2graph

        print('Start: structure graph')
        graph = array2graph(patterns, weight_type, edges='upper_right_triangle')
        print('Finish: structure graph')
    else:
        graph = None
    # -----------------------

    print('Start: calculate assessments')
    # -----------------------
    labels_list = []
    labels_files = [pjoin(result_dir, item) for item in os.listdir(result_dir)
                    if 'group_labels' in item]
    for labels_file in labels_files:
        labels_list.append(np.array(open(labels_file).read().split(' '), dtype=np.uint16))
    labels_list.sort(key=lambda x: len(set(x)))
    n_clusters = np.array([len(set(labels)) for labels in labels_list])
    indices = np.where(n_clusters > max_n_cluster)[0]
    if len(indices):
        end_idx = indices[0]
        labels_list = labels_list[:end_idx]
        n_clusters = n_clusters[:end_idx]
    n_labels = len(labels_list)
    for idx, labels in enumerate(labels_list):
        labels_uniq = np.unique(labels)
        for metric in assessments_dict.keys():
            if metric == 'dice':
                from commontool.algorithm.tool import calc_overlap

                sub_dices = []
                for label in labels_uniq:
                    subgroup_activ_roi = np.atleast_2d(activ_roi[labels == label])
                    subgroup_activ_roi_mean = np.mean(subgroup_activ_roi, 0)

                    collection1 = np.where(subgroup_activ_roi_mean > 2.3)[0]
                    collection2s = [np.where(i > 2.3)[0] for i in subgroup_activ_roi]
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
                    assessments_dict[metric].append(silhouette_score(patterns, labels,
                                                                     metric=weight_type[1], random_state=0))

            elif 'elbow' in metric:
                tmp = metric.split('_')
                assessment = elbow_score(patterns, labels, metric=weight_type[1],
                                         type=(tmp[1], tmp[2]))
                assessments_dict[metric].append(assessment)

            elif metric == 'gap statistic':
                pass

            else:
                raise RuntimeError("{} isn't a valid assessment metric".format(metric))

        print('Assessment calculated: {0}/{1}'.format(idx + 1, n_labels))

    if 'gap statistic' in assessments_dict.keys():
        from commontool.algorithm.cluster import hac_scipy
        from FFA_pattern.tool import k_means, gap_stat_mine

        if 'HAC' in clustering_method:
            cluster_method = hac_scipy
        elif 'KM' in clustering_method:
            cluster_method = k_means
        else:
            raise RuntimeError("analysis-{} isn't supported at present!".format(clustering_method))

        labels_list, gaps, s, k_selected = gap_stat_mine(patterns, n_clusters,
                                                         cluster_method=cluster_method)
        assessments_dict['gap statistic'] = (gaps, s, k_selected)

    x = np.arange(n_labels)
    x_labels = n_clusters
    vline_plotter_holder = []
    for metric_pair in assessment_metric_pairs:
        # plot assessment curve
        v_plotter = ClusteringVlineMoverPlotter(patterns, labels_list, meth_dir, result_dir)

        if metric_pair[0] == 'dice':
            y = np.mean(assessments_dict[metric_pair[0]], 1)
            sem = stats.sem(assessments_dict[metric_pair[0]], 1)
            v_plotter.axes[0].plot(x, y, 'b.-')
            v_plotter.axes[0].fill_between(x, y-sem, y+sem, alpha=0.5)

        elif 'elbow' in metric_pair[0]:
            y = assessments_dict[metric_pair[0]]
            v_plotter.axes[0].plot(x, y, 'k.-')

            x1 = x[:-1]
            y1 = [y[i] - y[i + 1] for i in x1]
            fig1, ax1 = plt.subplots()
            ax1.plot(x1, y1, 'k.-')
            # ax1.set_title('assessment for #subgroup')
            # ax1.set_xlabel('#subgroup')
            ax1.set_xlabel('k')
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
            ax2.plot(x2, y2, 'k.-')
            # ax2.set_title('assessment for #subgroups')
            # ax2.set_xlabel('#subgroups')
            # ax2.set_ylabel(metric_pair[0] + "''")
            ax2.set_xlabel('k')
            ax2.set_ylabel('-\u25b3V\u2096')
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

        # v_plotter.axes[0].set_title('assessment for #subgroups')
        # v_plotter.axes[0].set_xlabel('#subgroups')
        v_plotter.axes[0].set_xlabel('k')
        if n_labels > 2:
            if metric_pair[0] == 'gap statistic':
                vline_idx = np.where(x_labels == assessments_dict[metric_pair[0]][2])[0][0]
            else:
                vline_idx = int(n_labels / 2)
            v_plotter.axes[0].set_xticks(x[[0, vline_idx, -1]])
            v_plotter.axes[0].set_xticklabels(x_labels[[0, vline_idx, -1]])
            # plt.setp(v_plotter.axes[0].get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
        else:
            if metric_pair[0] == 'gap statistic':
                vline_idx = np.where(x_labels == assessments_dict[metric_pair[0]][2])[0][0]
            else:
                vline_idx = 0
            v_plotter.axes[0].set_xticks(x)
            v_plotter.axes[0].set_xticklabels(x_labels)
        # v_plotter.axes[0].set_ylabel(metric_pair[0], color='b')
        # v_plotter.axes[0].tick_params('y', colors='b')
        v_plotter.axes[0].set_ylabel('W\u2096')

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


def gen_mean_activation():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import save2nifti, CiftiReader

    # predefine some variates
    # -----------------------
    # predefine parameters
    prob_thr = 1.65
    n_clusters = [100]
    hemi = 'rh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    stats_table_titles = ['label', '#subject',
                          'min', 'max', 'mean',
                          'min_roi', 'max_roi', 'mean_roi']

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    clustering_dir = pjoin(proj_dir, 'analysis/s2/lh')
    n_cluster_dirs = pjoin(clustering_dir, 'zscore/HAC_ward_euclidean/{}clusters')
    roi_file = pjoin(proj_dir, f'data/HCP/label/MMPprob_OFA_FFA_thr1_{hemi}.label')
    activ_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    pattern_file = pjoin(proj_dir, 'analysis/s2/{}/zscore/roi_patterns.npy'.format(hemi))
    # -----------------------

    # get data
    roi = nib.freesurfer.read_label(roi_file)
    # activ = nib.load(activ_file).get_data().squeeze().T
    activ = CiftiReader(activ_file).get_data(brain_structure[hemi], True)
    patterns = np.load(pattern_file)

    # analyze labels
    # --------------
    for n_cluster in n_clusters:
        # get clustering labels of subjects
        n_cluster_dir = n_cluster_dirs.format(n_cluster)
        group_labels_file = pjoin(n_cluster_dir, 'group_labels')
        with open(group_labels_file) as rf:
            group_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        activation_dir = pjoin(n_cluster_dir, 'activation')
        if not os.path.exists(activation_dir):
            os.makedirs(activation_dir)

        stats_table_content = dict()
        for title in stats_table_titles:
            # initialize statistics table content
            stats_table_content[title] = []

        mean_activ = np.zeros((0, activ.shape[1]))
        mean_std_activ = np.zeros((0, activ.shape[1]))
        prob_activ = np.zeros((0, activ.shape[1]))
        mean_patterns = np.zeros((0, activ.shape[1]))
        mean_std_patterns = np.zeros((0, activ.shape[1]))
        for label in sorted(set(group_labels)):
            # get subgroup data
            subgroup_activ = np.atleast_2d(activ[group_labels == label])
            subgroup_activ_mean = np.mean(subgroup_activ, 0)
            subgroup_activ_prob = np.mean(subgroup_activ > prob_thr, 0)
            subgroup_activ_mean_std = subgroup_activ_mean / np.std(subgroup_activ, 0)
            subgroup_activ_mean_std[np.isnan(subgroup_activ_mean_std)] = 0
            subgroup_roi_activ_mean = subgroup_activ_mean[roi]

            mean_activ = np.r_[mean_activ, np.atleast_2d(subgroup_activ_mean)]
            mean_std_activ = np.r_[mean_std_activ, np.atleast_2d(subgroup_activ_mean_std)]
            prob_activ = np.r_[prob_activ, np.atleast_2d(subgroup_activ_prob)]

            stats_table_content['label'].append(str(label))
            stats_table_content['#subject'].append(str(subgroup_activ.shape[0]))
            stats_table_content['min'].append(str(np.min(subgroup_activ_mean)))
            stats_table_content['max'].append(str(np.max(subgroup_activ_mean)))
            stats_table_content['mean'].append(str(np.mean(subgroup_activ_mean)))
            stats_table_content['min_roi'].append(str(np.min(subgroup_roi_activ_mean)))
            stats_table_content['max_roi'].append(str(np.max(subgroup_roi_activ_mean)))
            stats_table_content['mean_roi'].append(str(np.mean(subgroup_roi_activ_mean)))

            # get mean patterns
            subgroup_patterns = patterns[group_labels == label]
            subgroup_patterns_mean = np.mean(subgroup_patterns, 0)
            subgroup_patterns_mean_std = subgroup_patterns_mean / np.std(subgroup_patterns, 0)
            subgroup_patterns_mean = subgroup_patterns_mean[None, :]
            subgroup_patterns_mean_std = subgroup_patterns_mean_std[None, :]

            pattern_mean_map = np.ones((1, activ.shape[1])) * np.min(subgroup_patterns_mean)
            pattern_mean_map[:, roi] = subgroup_patterns_mean
            mean_patterns = np.r_[mean_patterns, pattern_mean_map]

            pattern_mean_std_map = np.ones((1, activ.shape[1])) * np.min(subgroup_patterns_mean_std)
            pattern_mean_std_map[:, roi] = subgroup_patterns_mean_std
            mean_std_patterns = np.r_[mean_std_patterns, pattern_mean_std_map]

            # save2nifti(pjoin(activation_dir, f'activ_g{label}_{hemi}.nii.gz'), subgroup_activ.T[:, None, None, :])

        # output activ
        save2nifti(pjoin(activation_dir, f'mean_activ_{hemi}.nii.gz'), mean_activ.T[:, None, None, :])
        save2nifti(pjoin(activation_dir, f'mean_std_activ_{hemi}.nii.gz'), mean_std_activ.T[:, None, None, :])
        save2nifti(pjoin(activation_dir, f'prob{prob_thr}_activ_{hemi}.nii.gz'), prob_activ.T[:, None, None, :])
        save2nifti(pjoin(activation_dir, f'mean_patterns_{hemi}.nii.gz'), mean_patterns.T[:, None, None, :])
        save2nifti(pjoin(activation_dir, f'mean_std_patterns_{hemi}.nii.gz'), mean_std_patterns.T[:, None, None, :])

        # output statistics
        with open(pjoin(activation_dir, f'statistics_{hemi}.csv'), 'w') as f:
            f.write(','.join(stats_table_titles) + '\n')
            lines = []
            for title in stats_table_titles:
                lines.append(stats_table_content[title])
            for line in zip(*lines):
                f.write(','.join(line) + '\n')

        print('{}clusters: done'.format(n_cluster))


def gen_mean_structure():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import save2nifti

    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = [100]
    hemi = 'rh'

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    clustering_dir = pjoin(proj_dir, 'analysis/s2/lh')
    n_cluster_dirs = pjoin(clustering_dir, 'zscore/HAC_ward_euclidean/{}clusters')
    data_file = pjoin(proj_dir, 'analysis/s2/{}/curvature.nii.gz'.format(hemi))
    # -----------------------

    # get data
    data = nib.load(data_file).get_data().squeeze().T

    # analyze labels
    # --------------
    for n_cluster in n_clusters:
        # get clustering labels of subjects
        n_cluster_dir = n_cluster_dirs.format(n_cluster)
        group_labels_file = pjoin(n_cluster_dir, 'group_labels')
        with open(group_labels_file) as rf:
            group_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        structure_dir = pjoin(n_cluster_dir, 'structure')
        if not os.path.exists(structure_dir):
            os.makedirs(structure_dir)

        mean_data = np.zeros((0, data.shape[1]))
        for label in np.unique(group_labels):
            # get subgroup data
            subgroup_data = np.atleast_2d(data[group_labels == label])
            subgroup_data_mean = np.mean(subgroup_data, 0)
            mean_data = np.r_[mean_data, np.atleast_2d(subgroup_data_mean)]

        # output
        save2nifti(pjoin(structure_dir, f'mean_curv_{hemi}.nii.gz'), mean_data.T[:, None, None, :])
        print('{}clusters: done'.format(n_cluster))


if __name__ == '__main__':
    # get_roi_pattern()
    # clustering()
    # assess_n_cluster()
    gen_mean_activation()
    gen_mean_structure()
