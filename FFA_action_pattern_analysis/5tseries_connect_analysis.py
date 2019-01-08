import os
import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy.stats import ttest_ind, sem
from matplotlib import pyplot as plt
from commontool.io.io import save2nifti, CiftiReader, CsvReader
from commontool.algorithm.plot import auto_bar_width


def calc_mean_map(connect_files, items):
    connect_dir = os.path.dirname(connect_files)
    mean_dir = pjoin(connect_dir, 'mean')
    if not os.path.exists(mean_dir):
        os.makedirs(mean_dir)
    for item in items:
        connect_file = connect_files.format(item=item)
        data = np.atleast_2d(nib.load(connect_file).get_data())
        mean = np.mean(data, 0)

        out_name = os.path.basename(connect_file)
        save2nifti(pjoin(mean_dir, out_name), mean)


def calc_mean_sem(connect_files, items, roi_map_file, labels_l=None, labels_r=None):
    """
    calculate mean and sem across subjects

    :param connect_files:
    :param items:
    :param roi_map_file:
    :param labels_l:
    :param labels_r:
    :return:
    """
    connect_dir = os.path.dirname(connect_files)
    mean_sem_dir = pjoin(connect_dir, 'mean_sem')
    if not os.path.exists(mean_sem_dir):
        os.makedirs(mean_sem_dir)

    dict_lr = dict()
    if roi_map_file is not None:
        reader = CiftiReader(roi_map_file)
        roi_map_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
        roi_map_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)

        if labels_l is None:
            labels_l = [int(label) for label in np.unique(roi_map_l) if label != 0]
        else:
            labels_l = [int(label) for label in labels_l if label != 0]
        if labels_r is None:
            labels_r = [int(label) for label in np.unique(roi_map_r) if label != 0]
        else:
            labels_r = [int(label) for label in labels_r if label != 0]

        dict_lr['trg_vertices_l'] = [np.where(roi_map_l == i)[1] for i in labels_l]
        dict_lr['trg_vertices_r'] = [np.where(roi_map_r == i)[1] for i in labels_r]
        dict_lr['label_name_l'] = [reader.label_tables()[0][i].label for i in labels_l]
        dict_lr['label_name_r'] = [reader.label_tables()[0][i].label for i in labels_r]
        dict_lr['label_l'] = [str(i) for i in labels_l]
        dict_lr['label_r'] = [str(i) for i in labels_r]

    for item in items:
        connect_file = connect_files.format(item=item)
        data = np.atleast_2d(nib.load(connect_file).get_data())
        if dict_lr:
            trg_vertices_list = dict_lr['trg_vertices_' + item[-2]]
            label_names = dict_lr['label_name_' + item[-2]].copy()
            labels = dict_lr['label_' + item[-2]].copy()
        else:
            trg_vertices_list = [[vtx] for vtx in range(data.shape[1])]
            label_names = list(range(data.shape[1]))
            labels = list(range(data.shape[1]))
        means = ['mean']
        sems = ['sem']
        for trg_vertices in trg_vertices_list:
            trg_data = np.mean(data[:, trg_vertices], 1)
            means.append(str(np.mean(trg_data)))
            sems.append(str(sem(trg_data)))
        label_names.insert(0, 'label_name')
        labels.insert(0, 'label')

        line1 = ','.join(label_names)
        line2 = ','.join(means)
        line3 = ','.join(sems)
        line4 = ','.join(labels)
        lines = '\n'.join([line1, line2, line3, line4])
        open(pjoin(mean_sem_dir, item), 'w+').writelines(lines)


def plot_mean_sem(mean_sem_files, items, labels=None):

    fig, ax = plt.subplots()
    x = None
    width = None
    xticklabels = None
    rects_list = []
    item_num = len(items)
    for idx, item in enumerate(items):
        mean_sem_file = mean_sem_files.format(item)
        mean_sem_dict = CsvReader(mean_sem_file).to_dict(1)
        if labels is None:
            labels = mean_sem_dict['label']
        else:
            labels = [str(i) for i in labels]
        if x is None:
            xticklabels = [mean_sem_dict['label_name'][mean_sem_dict['label'].index(i)] for i in labels]
            x = np.arange(len(labels))
            width = auto_bar_width(x, item_num)
        y = [float(mean_sem_dict['mean'][mean_sem_dict['label'].index(i)]) for i in labels]
        sems = [float(mean_sem_dict['sem'][mean_sem_dict['label'].index(i)]) for i in labels]
        rects = ax.bar(x+width*idx, y, width, color='k', alpha=1./((idx+1)/2+0.5), yerr=sems)
        rects_list.append(rects)
    ax.legend(rects_list, items)
    ax.set_xticks(x+width/2.0*(item_num-1))
    ax.set_xticklabels(xticklabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('pearson r')
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


def compare(connect_files, item_pairs, roi_map_file=None, p_thr=1.0,
            labels_l=None, labels_r=None):

    connect_dir = os.path.dirname(connect_files)
    compare_dir = pjoin(connect_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    dict_lr = dict()
    if roi_map_file is not None:
        reader = CiftiReader(roi_map_file)
        roi_map_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
        roi_map_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)

        if labels_l is None:
            labels_l = [int(label) for label in np.unique(roi_map_l) if label != 0]
        else:
            labels_l = [int(label) for label in labels_l if label != 0]
        if labels_r is None:
            labels_r = [int(label) for label in np.unique(roi_map_r) if label != 0]
        else:
            labels_r = [int(label) for label in labels_r if label != 0]

        dict_lr['trg_vertices_l'] = [np.where(roi_map_l == i)[1] for i in labels_l]
        dict_lr['trg_vertices_r'] = [np.where(roi_map_r == i)[1] for i in labels_r]
        dict_lr['label_name_l'] = [reader.label_tables()[0][i].label for i in labels_l]
        dict_lr['label_name_r'] = [reader.label_tables()[0][i].label for i in labels_r]
        dict_lr['label_l'] = [str(i) for i in labels_l]
        dict_lr['label_r'] = [str(i) for i in labels_r]

    for item0, item1 in item_pairs:
        assert item0[-2] == item1[-2]
        data0 = np.atleast_2d(nib.load(connect_files.format(item=item0)).get_data())
        data1 = np.atleast_2d(nib.load(connect_files.format(item=item1)).get_data())
        assert data0.shape[1] == data1.shape[1]

        if dict_lr:
            trg_vertices_list = dict_lr['trg_vertices_' + item0[-2]]
            label_names = dict_lr['label_name_' + item0[-2]].copy()
            labels = dict_lr['label_' + item0[-2]].copy()
        else:
            trg_vertices_list = [[vtx] for vtx in range(data0.shape[1])]
            label_names = list(range(data0.shape[1]))
            labels = list(range(data0.shape[1]))

        ts = ['t']
        ps = ['p']
        compare_map = np.zeros(data0.shape[1])
        for trg_vertices in trg_vertices_list:
            trg_data0 = data0[:, trg_vertices]
            trg_data1 = data1[:, trg_vertices]
            t, p = ttest_ind(np.mean(trg_data0, 1), np.mean(trg_data1, 1))
            ts.append(str(t))
            ps.append(str(p))
            compare_map[trg_vertices] = t if p < p_thr else 0
        label_names.insert(0, 'label_name')
        labels.insert(0, 'label')

        if p_thr == 1:
            out_name = pjoin(compare_dir, '{}_vs_{}.nii.gz'.format(item0, item1))
            line1 = ','.join(label_names)
            line2 = ','.join(ts)
            line3 = ','.join(ps)
            line4 = ','.join(labels)
            lines = '\n'.join([line1, line2, line3, line4])
            open(pjoin(compare_dir, '{}_vs_{}'.format(item0, item1)), 'w+').writelines(lines)
        else:
            out_name = pjoin(compare_dir, '{}_vs_{}_p{}.nii.gz'.format(item0, item1, p_thr))
        save2nifti(out_name, compare_map)


def plot_compare(compare_file, labels=None, p_thr=None):

    file_name = os.path.basename(compare_file)
    compare_dict = CsvReader(compare_file).to_dict(1)
    if labels is None:
        labels = compare_dict['label']
    else:
        labels = [str(i) for i in labels]

    if p_thr is not None:
        labels = [i for i in labels if float(compare_dict['p'][compare_dict['label'].index(i)]) < p_thr]

    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = auto_bar_width(x, 2)
        y_t = [float(compare_dict['t'][compare_dict['label'].index(i)]) for i in labels]
        y_p = [float(compare_dict['p'][compare_dict['label'].index(i)]) for i in labels]
        rects_t = ax.bar(x, y_t, width, color='b', alpha=0.5)
        rects_p = ax_twin.bar(x, y_p, width, color='g', alpha=0.5)
        ax.legend([rects_t, rects_p], ['t', 'p'])
        ax.set_xticks(x)
        xticklabels = [compare_dict['label_name'][compare_dict['label'].index(i)] for i in labels]
        ax.set_xticklabels(xticklabels)
        plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(file_name)
    ax.set_ylabel('t', color='b')
    ax.tick_params('y', colors='b')
    ax_twin.set_ylabel('p', color='g')
    ax_twin.tick_params('y', colors='g')
    ax_twin.axhline(0.05)
    ax_twin.axhline(0.01)
    ax_twin.axhline(0.001)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/glasser_mmp')
    mask_roi_file_l = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/mmp_prob2.3_25_intersect_l.nii.gz')
    mask_roi_file_r = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/mmp_prob2.3_25_intersect_r.nii.gz')
    mask_rois_l = nib.load(mask_roi_file_l).get_data()
    mask_rois_r = nib.load(mask_roi_file_r).get_data()
    labels_l = [int(label) for label in np.unique(mask_rois_l) if label != 0]
    labels_r = [int(label) for label in np.unique(mask_rois_r) if label != 0]
    labels_dict = {
        'l': labels_l,
        'r': labels_r
    }
    print(len(labels_l))
    print(len(labels_r))

    # items = ['r1_FFA1_connect_l1', 'r1_FFA1_connect_r1',
    #          'r2_FFA1_connect_l2', 'r2_FFA1_connect_r2',
    #          'r1_FFA2_connect_l1', 'r1_FFA2_connect_r1']
    # calc_mean_map(connect_files, items)

    # items = ['r1_FFA1_connect_l1', 'r1_FFA1_connect_l2', 'r1_FFA1_connect_r1', 'r1_FFA1_connect_r2',
    #          'r1_FFA2_connect_l1', 'r1_FFA2_connect_l2', 'r1_FFA2_connect_r1', 'r1_FFA2_connect_r2',
    #          'r2_FFA1_connect_l2', 'r2_FFA1_connect_l1', 'r2_FFA1_connect_r2', 'r2_FFA1_connect_r1',
    #          'l1_FFA1_connect_l1', 'l1_FFA1_connect_l2', 'l1_FFA1_connect_r1', 'l1_FFA1_connect_r2',
    #          'l1_FFA2_connect_l1', 'l1_FFA2_connect_l2', 'l1_FFA2_connect_r1', 'l1_FFA2_connect_r2',
    #          'l2_FFA1_connect_l2', 'l2_FFA1_connect_l1', 'l2_FFA1_connect_r2', 'l2_FFA1_connect_r1']
    # calc_mean_sem(pjoin(connect_dir, '{item}.nii.gz'), items,
    #               '/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')

    # intersubgroup_items_list = [
    #     ['l1_FFA1_connect_l1', 'l1_FFA1_connect_l2'],
    #     ['l1_FFA1_connect_r1', 'l1_FFA1_connect_r2'],
    #     ['r1_FFA1_connect_l1', 'r1_FFA1_connect_l2'],
    #     ['r1_FFA1_connect_r1', 'r1_FFA1_connect_r2'],
    #     ['l1_FFA2_connect_l1', 'l1_FFA2_connect_l2'],
    #     ['l1_FFA2_connect_r1', 'l1_FFA2_connect_r2'],
    #     ['r1_FFA2_connect_l1', 'r1_FFA2_connect_l2'],
    #     ['r1_FFA2_connect_r1', 'r1_FFA2_connect_r2'],
    #     ['l2_FFA1_connect_l2', 'l2_FFA1_connect_l1'],
    #     ['l2_FFA1_connect_r2', 'l2_FFA1_connect_r1'],
    #     ['r2_FFA1_connect_l2', 'r2_FFA1_connect_l1'],
    #     ['r2_FFA1_connect_r2', 'r2_FFA1_connect_r1']
    # ]
    # interroi_items_list = [
    #     ['l1_FFA1_connect_l1', 'l2_FFA1_connect_l1', 'l1_FFA2_connect_l1'],
    #     ['l1_FFA1_connect_r1', 'l2_FFA1_connect_r1', 'l1_FFA2_connect_r1'],
    #     ['r1_FFA1_connect_l1', 'r2_FFA1_connect_l1', 'r1_FFA2_connect_l1'],
    #     ['r1_FFA1_connect_r1', 'r2_FFA1_connect_r1', 'r1_FFA2_connect_r1'],
    #     ['l1_FFA1_connect_l2', 'l2_FFA1_connect_l2', 'l1_FFA2_connect_l2'],
    #     ['l1_FFA1_connect_r2', 'l2_FFA1_connect_r2', 'l1_FFA2_connect_r2'],
    #     ['r1_FFA1_connect_l2', 'r2_FFA1_connect_l2', 'r1_FFA2_connect_l2'],
    #     ['r1_FFA1_connect_r2', 'r2_FFA1_connect_r2', 'r1_FFA2_connect_r2']
    # ]
    # for items in interroi_items_list:
    #     plot_mean_sem(pjoin(connect_dir, 'mean_sem/{}'), items, labels_dict[items[0][-2]])

    # intersubgroup_item_pairs = [
    #     ['l1_FFA1_connect_l1', 'l1_FFA1_connect_l2'],
    #     ['l1_FFA1_connect_r1', 'l1_FFA1_connect_r2'],
    #     ['r1_FFA1_connect_l1', 'r1_FFA1_connect_l2'],
    #     ['r1_FFA1_connect_r1', 'r1_FFA1_connect_r2'],
    #     ['l1_FFA2_connect_l1', 'l1_FFA2_connect_l2'],
    #     ['l1_FFA2_connect_r1', 'l1_FFA2_connect_r2'],
    #     ['r1_FFA2_connect_l1', 'r1_FFA2_connect_l2'],
    #     ['r1_FFA2_connect_r1', 'r1_FFA2_connect_r2'],
    #     ['l2_FFA1_connect_l2', 'l2_FFA1_connect_l1'],
    #     ['l2_FFA1_connect_r2', 'l2_FFA1_connect_r1'],
    #     ['r2_FFA1_connect_l2', 'r2_FFA1_connect_l1'],
    #     ['r2_FFA1_connect_r2', 'r2_FFA1_connect_r1']
    # ]
    interroi_item_pairs = [
        ['l2_FFA1_connect_l1', 'l1_FFA1_connect_l1'],
        ['l2_FFA1_connect_l1', 'l1_FFA2_connect_l1'],
        ['l2_FFA1_connect_r1', 'l1_FFA1_connect_r1'],
        ['l2_FFA1_connect_r1', 'l1_FFA2_connect_r1'],
        ['r2_FFA1_connect_l1', 'r1_FFA1_connect_l1'],
        ['r2_FFA1_connect_l1', 'r1_FFA2_connect_l1'],
        ['r2_FFA1_connect_r1', 'r1_FFA1_connect_r1'],
        ['r2_FFA1_connect_r1', 'r1_FFA2_connect_r1'],
        ['l2_FFA1_connect_l2', 'l1_FFA1_connect_l2'],
        ['l2_FFA1_connect_l2', 'l1_FFA2_connect_l2'],
        ['l2_FFA1_connect_r2', 'l1_FFA1_connect_r2'],
        ['l2_FFA1_connect_r2', 'l1_FFA2_connect_r2'],
        ['r2_FFA1_connect_l2', 'r1_FFA1_connect_l2'],
        ['r2_FFA1_connect_l2', 'r1_FFA2_connect_l2'],
        ['r2_FFA1_connect_r2', 'r1_FFA1_connect_r2'],
        ['r2_FFA1_connect_r2', 'r1_FFA2_connect_r2']
    ]
    # compare(pjoin(connect_dir, '{item}.nii.gz'), interroi_item_pairs,
    #         '/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')

    for item0, item1 in interroi_item_pairs:
        plot_compare(pjoin(connect_dir, 'compare/{}_vs_{}'.format(item0, item1)), labels_dict[item0[-2]], 0.05)
