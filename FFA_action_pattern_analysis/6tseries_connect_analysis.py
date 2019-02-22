import numpy as np


def get_samples(connectivity_data, seed_roi, all_rois, exclude_rois):

    seed_idx = all_rois.index(seed_roi)
    samples = list()
    sample_names = list()
    for trg_idx, roi in enumerate(all_rois):
        if roi not in exclude_rois:
            samples.append(connectivity_data[:, seed_idx, trg_idx])
            sample_names.append(roi)
    return samples, sample_names


if __name__ == '__main__':
    import os

    from os.path import join as pjoin
    from statsmodels.stats.multitest import multipletests  # https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    from matplotlib import pyplot as plt
    from commontool.io.io import CsvReader
    from commontool.algorithm.statistics import calc_mean_sem, plot_mean_sem, ttest_ind_pairwise, plot_compare

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/PAM_z165_p025_ROI')
    connectivity_file = pjoin(connect_dir, 'connectivity.npy')
    group_labels_file = pjoin(connect_dir, 'group_labels_4run_1200')
    npy_info_file = pjoin(connect_dir, 'npy_info')

    multi_test_corrected = True
    alpha = 1.1
    connectivity_data = np.load(connectivity_file)
    FFA_rois = ['l1_FFA', 'l2_FFA', 'r1_FFA', 'r2_FFA']
    subFFA_rois = ['l1_FFA1', 'l1_FFA2', 'l2_FFA1', 'r1_FFA1', 'r1_FFA2', 'r2_FFA1']
    FFA2name = {
        'l1_FFA': 'L_FFA', 'l2_FFA': 'L_FFA',
        'r1_FFA': 'R_FFA', 'r2_FFA': 'R_FFA',
        'l1_FFA1': 'L_FFA1', 'l1_FFA2': 'L_FFA2', 'l2_FFA1': 'L_FFA1',
        'r1_FFA1': 'R_FFA1', 'r1_FFA2': 'R_FFA2', 'r2_FFA1': 'R_FFA1'
    }
    seed2exclude = {
        'l1_FFA': [r for r in FFA_rois if r != 'r1_FFA'] + subFFA_rois,
        'l2_FFA': [r for r in FFA_rois if r != 'r2_FFA'] + subFFA_rois,
        'r1_FFA': [r for r in FFA_rois if r != 'l1_FFA'] + subFFA_rois,
        'r2_FFA': [r for r in FFA_rois if r != 'l2_FFA'] + subFFA_rois
    }
    all_rois = CsvReader(npy_info_file).to_dict(1)['region_name']
    group_labels = np.array(open(group_labels_file).read().split(' '))

    # ---calculate mean sem start---
    mean_sem_dir = pjoin(connect_dir, 'mean_sem_new')
    if not os.path.exists(mean_sem_dir):
        os.makedirs(mean_sem_dir)
    items = ['l1_FFA_g1', 'l2_FFA_g2', 'r1_FFA_g1', 'r2_FFA_g2']
    for item in items:
        seed_roi = item[:-3]
        sub_connectivity_data = connectivity_data[group_labels == item[-1]]
        samples, sample_names = get_samples(sub_connectivity_data,
                                            seed_roi, all_rois, seed2exclude[seed_roi])
        for idx, sample_name in enumerate(sample_names):
            if sample_name in FFA2name.keys():
                sample_names[idx] = FFA2name[sample_name]
        output_file = pjoin(mean_sem_dir, item)
        calc_mean_sem(samples, output_file, sample_names)
    # ---calculate mean sem end---

    # ---plot mean sem start---
    items_list = [
        ['l1_FFA_g1', 'l2_FFA_g2'],
        ['r1_FFA_g1', 'r2_FFA_g2']
    ]
    for items in items_list:
        mean_sem_files = [pjoin(mean_sem_dir, item) for item in items]
        plot_mean_sem(mean_sem_files, items, ylabel='pearson r')
    # ---plot mean sem end---

    # ---compare start---
    compare_dir = pjoin(connect_dir, 'compare_new')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    item_pairs = [
        ['l1_FFA_g1', 'l2_FFA_g2'],
        ['r1_FFA_g1', 'r2_FFA_g2']
    ]
    for item1, item2 in item_pairs:
        seed_roi1 = item1[:-3]
        seed_roi2 = item2[:-3]
        samples1, sample_names1 = get_samples(connectivity_data[group_labels == item1[-1]],
                                              seed_roi1, all_rois, seed2exclude[seed_roi1])
        for idx, sample_name1 in enumerate(sample_names1):
            if sample_name1 in FFA2name.keys():
                sample_names1[idx] = FFA2name[sample_name1]
        samples2, sample_names2 = get_samples(connectivity_data[group_labels == item2[-1]],
                                              seed_roi2, all_rois, seed2exclude[seed_roi2])
        output_file = pjoin(compare_dir, '{}_vs_{}'.format(item1, item2))
        ttest_ind_pairwise(samples1, samples2, output_file, sample_names1)

    for item1, item2, in item_pairs:
        file_name = '{}_vs_{}'.format(item1, item2)
        compare_file = pjoin(compare_dir, file_name)
        compare_dict = CsvReader(compare_file).to_dict(1)
        ps = np.array(list(map(float, compare_dict['p'])))
        if multi_test_corrected:
            reject, ps, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'fdr_bh')
        sample_names = [name for idx, name in enumerate(compare_dict['sample_name']) if ps[idx] < alpha]
        ps = [p for p in ps if p < alpha]
        print('\n'.join(list(map(str, zip(sample_names, ps)))))
        plot_compare(ps, sample_names, title=file_name)
    # ---compare end---

    plt.show()
