
hemi2structure = {
    'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}


def save_mean_maps():
    import os
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine parameters
    cluster_nums = [2]
    hemi = 'lh'
    structure_name = 'curv'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    src_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_curvature_MSMAll_32k_fs_LR.dscalar.nii')
    cluster_num_dirs = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/{}clusters')

    brain_structure = hemi2structure[hemi]

    # prepare data
    reader = CiftiReader(src_file)
    maps = reader.get_data(brain_structure, True)

    for cluster_num in cluster_nums:
        # get clustering labels of subjects
        cluster_num_dir = cluster_num_dirs.format(cluster_num)
        group_labels_path = pjoin(cluster_num_dir, 'group_labels')
        with open(group_labels_path) as rf:
            group_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        mean_maps = np.zeros((0, maps.shape[1]))
        for label in sorted(set(group_labels)):
            subgroup_maps = maps[group_labels == label]
            subgroup_maps_mean = np.atleast_2d(np.mean(subgroup_maps, 0))
            mean_maps = np.r_[mean_maps, subgroup_maps_mean]

        # output
        out_dir = pjoin(cluster_num_dir, 'structure')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save2nifti(pjoin(out_dir, '{}_{}_mean_maps.nii.gz'.format(hemi, structure_name)), mean_maps)

        print('{}_{}clusters: done'.format(structure_name, cluster_num))


def calc_acti_stru_corr():
    import numpy as np
    import nibabel as nib
    import pickle

    from os.path import join as pjoin
    from scipy.stats.stats import pearsonr
    from commontool.io.io import CiftiReader

    label2Label = {1: 2, 2: 1}
    stru_name = 'thickness'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    acti_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    stru_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_thickness_MSMAll_32k_fs_LR.dscalar.nii')
    lh_mask_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/lFFA_25.label')
    rh_mask_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/rFFA_25.label')
    group_labels_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/group_labels')
    out_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/structure/acti_stru_corrs_{}.pkl'.format(stru_name))

    lh_mask = nib.freesurfer.read_label(lh_mask_file)
    rh_mask = nib.freesurfer.read_label(rh_mask_file)
    lh_acti = CiftiReader(acti_file).get_data(hemi2structure['lh'], True)[:, lh_mask]
    rh_acti = CiftiReader(acti_file).get_data(hemi2structure['rh'], True)[:, rh_mask]
    lh_stru = CiftiReader(stru_file).get_data(hemi2structure['lh'], True)[:, lh_mask]
    rh_stru = CiftiReader(stru_file).get_data(hemi2structure['rh'], True)[:, rh_mask]
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)
    group_labels_uniq = np.unique(group_labels)

    acti_stuc_corrs = dict()
    for label1 in group_labels_uniq:
        indices1 = group_labels == label1
        sub_lh_acti = lh_acti[indices1]
        sub_rh_acti = rh_acti[indices1]
        for label2 in group_labels_uniq:
            indices2 = group_labels == label2
            sub_lh_stru = lh_stru[indices2]
            sub_rh_stru = rh_stru[indices2]
            lh_corr_name = 'G{}_acti_corr_G{}_{}_lFFA'.format(label2Label[label1], label2Label[label2], stru_name)
            rh_corr_name = 'G{}_acti_corr_G{}_{}_rFFA'.format(label2Label[label1], label2Label[label2], stru_name)
            acti_stuc_corrs[lh_corr_name] = [pearsonr(x, y)[0] for x in sub_lh_acti for y in sub_lh_stru]
            acti_stuc_corrs[rh_corr_name] = [pearsonr(x, y)[0] for x in sub_rh_acti for y in sub_rh_stru]
            print('lh', label1, label2, pearsonr(np.mean(sub_lh_acti, 0), np.mean(sub_lh_stru, 0))[0])
            print('rh', label1, label2, pearsonr(np.mean(sub_rh_acti, 0), np.mean(sub_rh_stru, 0))[0])
    with open(out_file, 'wb') as wf:
        pickle.dump(acti_stuc_corrs, wf, -1)


def calc_mean_sem():
    import os
    import pickle
    import numpy as np
    import pandas as pd

    from os.path import join as pjoin
    from scipy.stats.stats import sem

    stru_name = 'thickness'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    stru_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/structure')
    corr_file = pjoin(stru_dir, 'acti_stru_corrs_{}.pkl'.format(stru_name))
    mean_sem_dir = pjoin(stru_dir, 'mean_sem')
    if not os.path.exists(mean_sem_dir):
        os.makedirs(mean_sem_dir)

    acti_stru_corrs = pickle.load(open(corr_file, 'rb'))
    corr_names = sorted(acti_stru_corrs.keys())
    means = [np.mean(acti_stru_corrs[name]) for name in corr_names]
    sems = [sem(acti_stru_corrs[name]) for name in corr_names]
    df = pd.DataFrame({
        'names': corr_names,
        'means': means,
        'sems': sems
    })
    df.to_csv(pjoin(mean_sem_dir, stru_name), index=False)


def plot_mean_sem():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from os.path import join as pjoin
    from commontool.algorithm.plot import auto_bar_width, show_bar_value

    stru_name = 'myelin'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    stru_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/structure')
    mean_sem_file = pjoin(stru_dir, 'mean_sem/{}'.format(stru_name))

    mean_sems = pd.read_csv(mean_sem_file)
    intra_inter_pairs = np.array([
        ['G1_acti_corr_G1_{}_lFFA'.format(stru_name), 'G1_acti_corr_G2_{}_lFFA'.format(stru_name)],
        ['G2_acti_corr_G2_{}_lFFA'.format(stru_name), 'G2_acti_corr_G1_{}_lFFA'.format(stru_name)],
        ['G1_acti_corr_G1_{}_rFFA'.format(stru_name), 'G1_acti_corr_G2_{}_rFFA'.format(stru_name)],
        ['G2_acti_corr_G2_{}_rFFA'.format(stru_name), 'G2_acti_corr_G1_{}_rFFA'.format(stru_name)]
    ])
    names = mean_sems['names'].to_list()
    means = list(map(float, mean_sems['means']))
    sems = list(map(float, mean_sems['sems']))

    fig, ax = plt.subplots()
    x = np.arange(intra_inter_pairs.shape[0])
    item_num = intra_inter_pairs.shape[1]
    width = auto_bar_width(x, item_num)
    for idx in range(item_num):
        sub_means = [means[names.index(i)] for i in intra_inter_pairs[:, idx]]
        sub_sems = [sems[names.index(i)] for i in intra_inter_pairs[:, idx]]
        rects = ax.bar(x + width * idx, sub_means, width, color='k', alpha=1. / ((idx + 1) / 2 + 0.5), yerr=sub_sems)
        show_bar_value(rects, '.3f')
    xticklabels1 = [name for name in intra_inter_pairs[:, 0]]
    xticklabels2 = [name for name in intra_inter_pairs[:, 1]]
    xticklabels = xticklabels1 + xticklabels2
    ax.set_xticks(np.r_[x, x + width * (item_num - 1)])
    ax.set_xticklabels(xticklabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('pearsonr')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


def compare():
    import os
    import pickle
    import numpy as np

    from os.path import join as pjoin
    from commontool.algorithm.statistics import ttest_ind_pairwise

    stru_name = 'thickness'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    stru_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/structure')
    corr_file = pjoin(stru_dir, 'acti_stru_corrs_{}.pkl'.format(stru_name))
    compare_dir = pjoin(stru_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    out_file = pjoin(compare_dir, stru_name)

    intra_inter_pairs = np.array([
        ['G1_acti_corr_G1_{}_lFFA'.format(stru_name), 'G1_acti_corr_G2_{}_lFFA'.format(stru_name)],
        ['G2_acti_corr_G2_{}_lFFA'.format(stru_name), 'G2_acti_corr_G1_{}_lFFA'.format(stru_name)],
        ['G1_acti_corr_G1_{}_rFFA'.format(stru_name), 'G1_acti_corr_G2_{}_rFFA'.format(stru_name)],
        ['G2_acti_corr_G2_{}_rFFA'.format(stru_name), 'G2_acti_corr_G1_{}_rFFA'.format(stru_name)]
    ])

    acti_stru_corrs = pickle.load(open(corr_file, 'rb'))
    samples1 = []
    samples2 = []
    sample_names = []
    for name1, name2 in zip(intra_inter_pairs[:, 0], intra_inter_pairs[:, 1]):
        sample_names.append(name1 + '_vs_' + name2)
        samples1.append(acti_stru_corrs[name1])
        samples2.append(acti_stru_corrs[name2])
    ttest_ind_pairwise(samples1, samples2, out_file, sample_names)


def compare_plot_bar():
    import numpy as np
    import matplotlib.pyplot as plt

    from os.path import join as pjoin
    # https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    from statsmodels.stats.multitest import multipletests
    from commontool.io.io import CsvReader
    from commontool.algorithm.plot import auto_bar_width, show_bar_value

    stru_name = 'myelin'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    stru_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/structure')
    compare_dir = pjoin(stru_dir, 'compare')
    compare_file = pjoin(compare_dir, stru_name)

    multi_test_corrected = True
    alpha = 0.001
    compare_dict = CsvReader(compare_file).to_dict(1)
    ps = np.array(list(map(float, compare_dict['p'])))
    if multi_test_corrected:
        reject, ps, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'fdr_bh')
    sample_names = [name for idx, name in enumerate(compare_dict['sample_name']) if ps[idx] < alpha]
    ps = [p for p in ps if p < alpha]
    print('\n'.join(list(map(str, zip(sample_names, ps)))))

    fig, ax = plt.subplots()
    x = np.arange(len(sample_names))
    width = auto_bar_width(x)
    rects_p = ax.bar(x, ps, width, color='g', alpha=0.5)
    show_bar_value(rects_p, '.2f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(stru_name)
    ax.set_ylabel('p', color='g')
    ax.tick_params('y', colors='g')
    ax.axhline(0.05)
    ax.axhline(0.01)
    ax.axhline(0.001)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # calc_acti_stru_corr()
    # calc_mean_sem()
    # plot_mean_sem()
    # compare()
    compare_plot_bar()
