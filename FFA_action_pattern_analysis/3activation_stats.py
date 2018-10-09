import numpy as np
import nibabel as nib

from collections import OrderedDict
from os.path import join as pjoin
from matplotlib import pyplot as plt
from tmp_tool.io import read_nifti
from commontool.io.io import CsvReader
from commontool.algorithm.plot import show_bar_value, auto_bar_width


def explore_statistics(n_clusters_dir, items, ylabels, colors, val_fmt=''):
    stats_path = pjoin(n_clusters_dir, 'statistics.csv')

    with open(stats_path) as f:
        stats = f.read().splitlines()
    stats_items = stats[0].split(',')
    stats_content = [line.split(',') for line in stats[1:]]
    stats_content = list(zip(*stats_content))
    stats_dict = {}
    for idx, item in enumerate(stats_items):
        stats_dict[item] = stats_content[idx]

    x = np.arange(len(stats) - 1)
    width = auto_bar_width(x)
    for idx, item in enumerate(items):
        plt.figure()
        y = [float(_) for _ in stats_dict[item]]
        rects = plt.bar(x, y, width, color=colors[idx])
        show_bar_value(rects, val_fmt)
        plt.xlabel('subgroup label')
        plt.ylabel(ylabels[idx])
        plt.title(item)
        plt.xticks(x, stats_dict['label'])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(pjoin(n_clusters_dir, '{}.png'.format(item)))


def explore_label_dice(n_clusters_dir):
    import nibabel as nib

    from commontool.algorithm.tool import calc_overlap

    c1_r = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster1_ROI_z2.3.label'))
    c2_r = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster2_ROI_z2.3.label'))
    c3_r = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster3_ROI_z2.3.label'))
    c4_r1 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster4_ROI1_z2.3.label'))
    c4_r2 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster4_ROI2_z2.3.label'))
    c5_r1 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster5_ROI1_z2.3.label'))
    c5_r2 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster5_ROI2_z2.3.label'))
    c6_r1 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster6_ROI1_z2.3.label'))
    c6_r2 = nib.freesurfer.read_label(pjoin(n_clusters_dir, 'cluster6_ROI2_z2.3.label'))
    c1_6_acti_top10 = nib.load(pjoin(n_clusters_dir, 'top_acti_ROIs_percent10.0.nii.gz')).get_data()

    c123_r_z = [c1_r, c2_r, c3_r]
    c123_r_top = c1_6_acti_top10[:3]
    c123_dice = []
    c123_xticks = ['c1_c2_z2.3', 'c1_c3_z2.3', 'c2_c3_z2.3',
                   'c1_c2_top10', 'c1_c3_top10', 'c2_c3_top10']

    c456_r_z = [np.concatenate((c4_r1, c4_r2)), np.concatenate((c5_r1, c5_r2)), np.concatenate((c6_r1, c6_r2))]
    c456_r_top = c1_6_acti_top10[3:]
    c456_dice = []
    c456_xticks = ['c4_c5_z2.3', 'c4_c6_z2.3', 'c5_c6_z2.3',
                   'c4_c5_top10', 'c4_c6_top10', 'c5_c6_top10']

    for idx, i in enumerate(c123_r_z[:-1]):
        for j in c123_r_z[idx + 1:]:
            c123_dice.append(calc_overlap(i, j))

    for idx, i in enumerate(c123_r_top[:-1]):
        for j in c123_r_top[idx + 1:]:
            c123_dice.append(calc_overlap(i, j, 1, 1))

    for idx, i in enumerate(c456_r_z[:-1]):
        for j in c456_r_z[idx + 1:]:
            c456_dice.append(calc_overlap(i, j))

    for idx, i in enumerate(c456_r_top[:-1]):
        for j in c456_r_top[idx + 1:]:
            c456_dice.append(calc_overlap(i, j, 1, 1))

    x = np.arange(6)
    width = auto_bar_width(x)
    plt.figure()
    rects = plt.bar(x, c123_dice, width, color='b')
    show_bar_value(rects, '.2%')
    plt.ylabel('dice')
    plt.xticks(x, c123_xticks)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.figure()
    rects = plt.bar(x, c456_dice, width, color='b')
    show_bar_value(rects, '.2%')
    plt.ylabel('dice')
    plt.xticks(x, c456_xticks)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def explore_roi_stats(n_clusters_dir):
    roi_path = pjoin(n_clusters_dir, 'mean_map_ROIs.nii.gz')
    stats_path = pjoin(n_clusters_dir, 'statistics.csv')
    roi_maps = read_nifti(roi_path)
    stats_reader = CsvReader(stats_path)
    row_dict = stats_reader.to_dict(keys=['#subjects'])
    
    numb_items = ['1', '2']
    numb_dict = OrderedDict()
    for item in numb_items:
        numb_dict[item] = 0
    
    type_items = ['r_pFFA', 'r_mFFA', 'both', 'unknown']
    type_dict = OrderedDict()
    for item in type_items:
        type_dict[item] = 0
    
    for idx, roi_map in enumerate(roi_maps):
        map_set = set(roi_map)
        subjects_num = int(row_dict['#subjects'][idx])
        
        if 0 not in map_set:
            raise RuntimeError('Be careful! There is no zero in one roi_map')
        
        if len(map_set) == 2:    
            numb_dict['1'] += subjects_num
            if 1 in map_set:
                type_dict['r_pFFA'] += subjects_num
            elif 2 in map_set:
                type_dict['r_mFFA'] += subjects_num
            elif 3 in map_set:
                type_dict['unknown'] += subjects_num
            else:
                raise RuntimeError('Be careful! the only one ROI label is not in (1, 2, 3)')
        elif len(map_set) == 3:
            numb_dict['2'] += subjects_num
            if 1 in map_set and 2 in map_set:
                type_dict['both'] += subjects_num
            else:
                raise RuntimeError('Be careful! the two ROI labels are not 1 and 2')
        else:
            raise RuntimeError('Be careful! the number of ROI labels is not 1 or 2')
    
    plt.figure()
    x = np.arange(len(numb_items))
    width = auto_bar_width(x)
    rects = plt.bar(x, numb_dict.values(), width, color='b')
    show_bar_value(rects)
    plt.ylabel('#subjects')
    plt.xticks(x, numb_dict.keys())
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(pjoin(n_clusters_dir, 'numb_count.png'))
    
    plt.figure()
    x = np.arange(len(type_items))
    width = auto_bar_width(x)
    rects = plt.bar(x, type_dict.values(), width, color='b')
    show_bar_value(rects)
    plt.ylabel('#subjects')
    plt.xticks(x, type_dict.keys())
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(pjoin(n_clusters_dir, 'type_count.png'))


if __name__ == '__main__':
    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = 10
    subproject_name = '2mm_ward_regress'

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))

    # explore_statistics(n_clusters_dir,
    #                    items=['#subjects', 'map_mean', 'rFFA_mean'],
    #                    ylabels=['count', 'z_stat', 'z_stat'],
    #                    colors=['r', 'g', 'b'])
    # explore_label_dice(n_clusters_dir)
    explore_roi_stats(n_clusters_dir)

    plt.show()
