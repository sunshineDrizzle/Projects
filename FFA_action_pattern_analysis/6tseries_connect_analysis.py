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
    # connectivity_file = pjoin(connect_dir, 'connectivity.npy')
    # group_labels_file = pjoin(connect_dir, 'group_labels_4run_1200')
    npy_info_file = pjoin(connect_dir, 'npy_info')

    multi_test_corrected = True
    alpha = 1.1
    # connectivity_data = np.load(connectivity_file)
    FFA_rois = ['l1_FFA', 'l2_FFA', 'r1_FFA', 'r2_FFA']
    subFFA_rois = ['l1_FFA1', 'l1_FFA2', 'l2_FFA1', 'r1_FFA1', 'r1_FFA2', 'r2_FFA1']
    FFA2name = {
        'l1_FFA': 'L_FFA', 'l2_FFA': 'L_FFA',
        'r1_FFA': 'R_FFA', 'r2_FFA': 'R_FFA',
        'l1_FFA1': 'L_FFA1', 'l1_FFA2': 'L_FFA2', 'l2_FFA1': 'L_FFA1',
        'r1_FFA1': 'R_FFA1', 'r1_FFA2': 'R_FFA2', 'r2_FFA1': 'R_FFA1'
    }
    FFA2name2 = {
        'l1_FFA': 'G2_lFFA', 'l2_FFA': 'G1_lFFA',
        'r1_FFA': 'G2_rFFA', 'r2_FFA': 'G1_rFFA',
        'l1_FFA1': 'G2_lFFA1', 'l1_FFA2': 'G2_lFFA2', 'l2_FFA1': 'G1_lFFA1',
        'r1_FFA1': 'G2_rFFA1', 'r1_FFA2': 'G2_rFFA2', 'r2_FFA1': 'G1_rFFA1'
    }
    item2exclude = {
        'l1_FFA': [r for r in FFA_rois if r != 'r1_FFA'] + subFFA_rois,
        'l2_FFA': [r for r in FFA_rois if r != 'r2_FFA'] + subFFA_rois,
        'r1_FFA': [r for r in FFA_rois if r != 'l1_FFA'] + subFFA_rois,
        'r2_FFA': [r for r in FFA_rois if r != 'l2_FFA'] + subFFA_rois,
        'L_OFA_g1': ['L_OFA', 'l2_FFA', 'r2_FFA'] + subFFA_rois,
        'L_OFA_g2': ['L_OFA', 'l1_FFA', 'r1_FFA'] + subFFA_rois,
        'R_OFA_g1': ['R_OFA', 'l2_FFA', 'r2_FFA'] + subFFA_rois,
        'R_OFA_g2': ['R_OFA', 'l1_FFA', 'r1_FFA'] + subFFA_rois
    }
    # all_rois = CsvReader(npy_info_file).to_dict(1)['region_name']
    # group_labels = np.array(open(group_labels_file).read().split(' '))

    # ---calculate mean sem start---
    # mean_sem_dir = pjoin(connect_dir, 'mean_sem_sess2')
    # if not os.path.exists(mean_sem_dir):
    #     os.makedirs(mean_sem_dir)
    # items = ['L_OFA_g1', 'L_OFA_g2', 'R_OFA_g1', 'R_OFA_g2']
    # items = FFA_rois
    # for item in items:
    #     seed_roi = item
    #     sub_connectivity_data = connectivity_data[group_labels == item[1]]
    #     samples, sample_names = get_samples(sub_connectivity_data,
    #                                         seed_roi, all_rois, item2exclude[item])
    #     for idx, sample_name in enumerate(sample_names):
    #         if sample_name in FFA2name.keys():
    #             sample_names[idx] = FFA2name[sample_name]
    #     output_file = pjoin(mean_sem_dir, item)
    #     calc_mean_sem(samples, output_file, sample_names)
    # ---calculate mean sem end---

    # ---plot mean sem start---
    # items_list = [
    #     ['L_OFA_g1', 'L_OFA_g2'],
    #     ['R_OFA_g1', 'R_OFA_g2']
    # ]
    # items_list = [
    #     ['l2_FFA', 'l1_FFA'],
    #     ['r2_FFA', 'r1_FFA']
    # ]
    # for items in items_list:
    #     mean_sem_files = [pjoin(mean_sem_dir, item) for item in items]
    #     name2s = [FFA2name2[item] for item in items]
    #     plot_mean_sem(mean_sem_files, name2s, ylabel='pearson r')
    # ---plot mean sem end---

    # ---compare start---
    compare_dir = pjoin(connect_dir, 'compare_sess2')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    # item_pairs = [
    #     ['L_OFA_g1', 'L_OFA_g2'],
    #     ['R_OFA_g1', 'R_OFA_g2']
    # ]
    item_pairs = [
        ['l1_FFA', 'l2_FFA'],
        ['r1_FFA', 'r2_FFA']
    ]
    # for item1, item2 in item_pairs:
    #     seed_roi1 = item1
    #     seed_roi2 = item2
    #     samples1, sample_names1 = get_samples(connectivity_data[group_labels == item1[1]],
    #                                           seed_roi1, all_rois, item2exclude[item1])
    #     for idx, sample_name1 in enumerate(sample_names1):
    #         if sample_name1 in FFA2name.keys():
    #             sample_names1[idx] = FFA2name[sample_name1]
    #     samples2, sample_names2 = get_samples(connectivity_data[group_labels == item2[1]],
    #                                           seed_roi2, all_rois, item2exclude[item2])
    #     output_file = pjoin(compare_dir, '{}_vs_{}'.format(item1, item2))
    #     ttest_ind_pairwise(samples1, samples2, output_file, sample_names1)

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
        # plot_compare(ps, sample_names, title=file_name)

        # ---20190605---
        ts = [float(t) for idx, t in enumerate(compare_dict['t']) if ps[idx] < alpha]
        ts_mat = np.zeros((7, 6))
        names_mat = np.zeros_like(ts_mat, np.object)
        ps_mat = np.ones_like(ts_mat)

        ffa_idx = 0
        for idx, name in enumerate(sample_names):
            if 'FFA' in name:
                ffa_idx = idx
                break
        ts.pop(ffa_idx)
        ps.pop(ffa_idx)
        sample_names.pop(ffa_idx)

        for i in range(7):
            for j in range(6):
                idx = i * 6 + j
                ts_mat[i, j] = ts[idx]
                names_mat[i, j] = sample_names[idx]
                ps_mat[i, j] = ps[idx]
        ts_mat[ps_mat > 0.05] = 0

        fig, ax = plt.subplots()
        im = ax.imshow(ts_mat, cmap='hot', vmin=0, vmax=7)
        for i in range(7):
            for j in range(6):
                if names_mat[i, j]:
                    if ts_mat[i, j] == 0:
                        c = 'w'
                    else:
                        c = 'k'
                    text = ax.text(j, i, names_mat[i, j], ha="center", va="center", color=c, fontsize=8)

        # plt.axis('off')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax.set_xticks([])
        ax.set_yticks([])
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.05)
        # cbar = fig.colorbar(im, ax=ax, cax=cax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('t')
        fig.tight_layout()
    # ---compare end---

    plt.show()
