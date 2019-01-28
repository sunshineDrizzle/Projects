import numpy as np


def get_samples(connectivity_file, seed_roi, all_rois, exclude_rois):

    connectivity_data = np.load(connectivity_file)
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
    connectivity_files = pjoin(connect_dir, 'subgroup{}_connectivity.npy')
    npy_info_file = pjoin(connect_dir, 'npy_info')

    ffa_rois = ['l1_FFA', 'l1_FFA1', 'l1_FFA2', 'l2_FFA', 'l2_FFA1',
                'r1_FFA', 'r1_FFA1', 'r1_FFA2', 'r2_FFA', 'r2_FFA1']
    all_rois = CsvReader(npy_info_file).to_dict(1)['region_name']

    # ---calculate mean sem start---
    mean_sem_dir = pjoin(connect_dir, 'mean_sem_test')
    if not os.path.exists(mean_sem_dir):
        os.makedirs(mean_sem_dir)
    seed_rois = ['l1_FFA', 'l2_FFA', 'r1_FFA', 'r2_FFA']
    exclude_rois = ffa_rois
    for seed_roi in seed_rois:
        samples, sample_names = get_samples(connectivity_files.format(seed_roi[1]),
                                            seed_roi, all_rois, exclude_rois)
        calc_mean_sem(samples, pjoin(mean_sem_dir, seed_roi), sample_names)
    # ---calculate mean sem end---

    # ---compare start---
    compare_dir = pjoin(connect_dir, 'compare_test')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    seed_roi_pairs = [
        ['l1_FFA', 'l2_FFA'],
        ['r1_FFA', 'r2_FFA']
    ]
    exclude_rois = ffa_rois
    for seed_roi1, seed_roi2 in seed_roi_pairs:
        samples1, sample_names = get_samples(connectivity_files.format(seed_roi1[1]),
                                             seed_roi1, all_rois, exclude_rois)
        samples2, _ = get_samples(connectivity_files.format(seed_roi2[1]),
                                  seed_roi2, all_rois, exclude_rois)
        output_file = pjoin(compare_dir, '{}_vs_{}'.format(seed_roi1, seed_roi2))
        ttest_ind_pairwise(samples1, samples2, output_file, sample_names)

    for seed_roi1, seed_roi2, in seed_roi_pairs:
        file_name = '{}_vs_{}'.format(seed_roi1, seed_roi2)
        compare_file = pjoin(compare_dir, file_name)
        compare_dict = CsvReader(compare_file).to_dict(1)
        p_uncorrected = np.array(list(map(float, compare_dict['p'])))
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_uncorrected, 0.05, 'fdr_bh')
        ts = list()
        ps = list()
        sample_names = list()
        for idx, sample_name in enumerate(compare_dict['sample_name']):
            if p_corrected[idx] < 1.1:
                ts.append(float(compare_dict['t'][idx]))
                ps.append(p_uncorrected[idx])
                sample_names.append(sample_name)
        print('\n'.join(list(map(str, zip(sample_names, ps)))))
        plot_compare(ps, sample_names, title=file_name)
    # ---compare end---

    # ---plot mean sem start---
    seed_rois_list = [
        ['l1_FFA', 'l2_FFA'],
        ['r1_FFA', 'r2_FFA']
    ]
    mean_sem_dir = pjoin(connect_dir, 'mean_sem_test')
    compare_dir = pjoin(connect_dir, 'compare_test')
    for seed_rois in seed_rois_list:
        compare_file = pjoin(compare_dir, '{}_vs_{}'.format(seed_rois[0], seed_rois[1]))
        compare_dict = CsvReader(compare_file).to_dict(1)
        p_uncorrected = np.array(list(map(float, compare_dict['p'])))
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_uncorrected, 0.05, 'fdr_bh')
        print('p_uncorrected:', p_uncorrected)
        print('p_corrected:', p_corrected)
        sample_names = compare_dict['sample_name']
        sample_names = [i for i in sample_names if p_corrected[compare_dict['sample_name'].index(i)] < 1]
        mean_sem_files = [pjoin(mean_sem_dir, seed_roi) for seed_roi in seed_rois]
        plot_mean_sem(mean_sem_files, seed_rois, sample_names, ylabel='pearson r')
    # ---plot mean sem end---

    plt.tight_layout()
    plt.show()
