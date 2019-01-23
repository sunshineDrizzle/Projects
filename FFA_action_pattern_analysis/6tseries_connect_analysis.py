import numpy as np

from scipy.stats import ttest_ind, sem
from matplotlib import pyplot as plt
from commontool.io.io import CsvReader
from commontool.algorithm.plot import auto_bar_width


def calc_mean_sem(samples, output_file, sample_names=None):
    """
    calculate mean and sem for each sample

    :param samples: sequence
        a sequence of samples
    :param output_file: str
    :param sample_names: sequence
        a sequence of sample names
    """
    if sample_names is None:
        sample_names = list(map(str, range(1, len(samples)+1)))
    else:
        assert len(samples) == len(sample_names)
    sample_names.insert(0, 'sample_name')

    means = ['mean']
    sems = ['sem']
    for sample in samples:
        means.append(str(np.nanmean(sample)))
        sems.append(str(sem(sample)))

    lines = list()
    lines.append(','.join(sample_names))
    lines.append(','.join(means))
    lines.append(','.join(sems))
    open(output_file, 'w+').writelines('\n'.join(lines))


def ttest_ind_pairwise(samples1, samples2, output_file, sample_names=None):
    """
    Do two sample t test pairwise between samples1 and samples2

    :param samples1: sequence
        a sequence of samples
    :param samples2: sequence
        a sequence of samples
    :param output_file: str
    :param sample_names: sequence
        a sequence of sample names
    """
    assert len(samples1) == len(samples2)
    sample_num = len(samples1)
    if sample_names is None:
        sample_names = list(map(str, range(1, sample_num+1)))
    else:
        assert len(sample_names) == sample_num
    sample_names.insert(0, 'sample_name')

    ts = ['t']
    ps = ['p']
    for idx in range(sample_num):
        sample1 = samples1[idx]
        sample2 = samples2[idx]
        t, p = ttest_ind(sample1, sample2)
        ts.append(str(t))
        ps.append(str(p))

    lines = list()
    lines.append(','.join(sample_names))
    lines.append(','.join(ts))
    lines.append(','.join(ps))
    open(output_file, 'w+').writelines('\n'.join(lines))


def plot_mean_sem(mean_sem_files, items=None, sample_names=None, xlabel='', ylabel=''):

    fig, ax = plt.subplots()
    x = None
    width = None
    xticklabels = None
    rects_list = []
    item_num = len(mean_sem_files)
    for idx, mean_sem_file in enumerate(mean_sem_files):
        mean_sem_dict = CsvReader(mean_sem_file).to_dict(1)
        if sample_names is None:
            sample_names = mean_sem_dict['sample_name']
        if x is None:
            xticklabels = sample_names
            x = np.arange(len(xticklabels))
            width = auto_bar_width(x, item_num)
        y = [float(mean_sem_dict['mean'][mean_sem_dict['sample_name'].index(i)]) for i in sample_names]
        sems = [float(mean_sem_dict['sem'][mean_sem_dict['sample_name'].index(i)]) for i in sample_names]
        rects = ax.bar(x+width*idx, y, width, color='k', alpha=1./((idx+1)/2+0.5), yerr=sems)
        rects_list.append(rects)
    if items is not None:
        assert item_num == len(items)
        ax.legend(rects_list, items)
    ax.set_xticks(x+width/2.0*(item_num-1))
    ax.set_xticklabels(xticklabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
    # plt.ylim(bottom=5.5)

    plt.tight_layout()
    plt.show()


def plot_compare(compare_file, sample_names=None, p_thr=None, title=''):

    compare_dict = CsvReader(compare_file).to_dict(1)
    if sample_names is None:
        sample_names = compare_dict['sample_name']

    if p_thr is not None:
        sample_names = [i for i in sample_names if float(compare_dict['p'][compare_dict['sample_name'].index(i)]) < p_thr]

    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    if len(sample_names) > 0:
        x = np.arange(len(sample_names))
        width = auto_bar_width(x)
        y_t = [float(compare_dict['t'][compare_dict['sample_name'].index(i)]) for i in sample_names]
        y_p = [float(compare_dict['p'][compare_dict['sample_name'].index(i)]) for i in sample_names]
        rects_t = ax.bar(x, y_t, width, color='b', alpha=0.5)
        rects_p = ax_twin.bar(x, y_p, width, color='g', alpha=0.5)
        ax.legend([rects_t, rects_p], ['t', 'p'])
        ax.set_xticks(x)
        xticklabels = sample_names
        ax.set_xticklabels(xticklabels)
        plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    ax.set_ylabel('t', color='b')
    ax.tick_params('y', colors='b')
    ax_twin.set_ylabel('p', color='g')
    ax_twin.tick_params('y', colors='g')
    ax_twin.axhline(0.05)
    ax_twin.axhline(0.01)
    ax_twin.axhline(0.001)

    plt.tight_layout()
    plt.show()


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
    from statsmodels.stats.multitest import multipletests
    # https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/PAM_z165_p025_ROI')
    connectivity_files = pjoin(connect_dir, 'subgroup{}_connectivity.npy')
    npy_info_file = pjoin(connect_dir, 'npy_info')

    ffa_rois = ['l1_FFA', 'l1_FFA1', 'l1_FFA2', 'l2_FFA', 'l2_FFA1',
                'r1_FFA', 'r1_FFA1', 'r1_FFA2', 'r2_FFA', 'r2_FFA1']
    all_rois = CsvReader(npy_info_file).to_dict(1)['region_name']

    # ---calculate mean sem start---
    mean_sem_dir = pjoin(connect_dir, 'mean_sem')
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
    compare_dir = pjoin(connect_dir, 'compare')
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
        plot_compare(compare_file, title=file_name)
    # ---compare end---

    # ---plot mean sem start---
    seed_rois_list = [
        ['l1_FFA', 'l2_FFA'],
        ['r1_FFA', 'r2_FFA']
    ]
    mean_sem_dir = pjoin(connect_dir, 'mean_sem')
    compare_dir = pjoin(connect_dir, 'compare')
    for seed_rois in seed_rois_list:
        compare_file = pjoin(compare_dir, '{}_vs_{}'.format(seed_rois[0], seed_rois[1]))
        compare_dict = CsvReader(compare_file).to_dict(1)
        p_uncorrected = np.array(list(map(float, compare_dict['p'])))
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_uncorrected, 0.05, 'fdr_bh')
        print('p_uncorrected:', p_uncorrected)
        print('p_corrected:', p_corrected)
        sample_names = compare_dict['sample_name']
        sample_names = [i for i in sample_names if float(p_corrected[compare_dict['sample_name'].index(i)]) < 0.05]
        mean_sem_files = [pjoin(mean_sem_dir, seed_roi) for seed_roi in seed_rois]
        plot_mean_sem(mean_sem_files, seed_rois, sample_names, ylabel='pearson r')
    # ---plot mean sem end---
