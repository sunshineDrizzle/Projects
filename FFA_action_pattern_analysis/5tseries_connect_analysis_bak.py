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


def calc_mean_sem(maps, output_file, mask=None, label_names=None):
    """
    calculate mean and sem across subjects
    """
    vtx_num = maps.shape[1]
    if mask is None:
        label_ids = list(map(str, range(vtx_num)))
        trg_vertices_list = [[vtx] for vtx in range(vtx_num)]
    else:
        assert mask.ndim == 1
        assert vtx_num == mask.shape[0]
        label_ids = [int(label_id) for label_id in np.unique(mask) if label_id != 0]
        trg_vertices_list = [np.where(mask == i)[0] for i in label_ids]
        label_ids = list(map(str, label_ids))
    if label_names is None:
        label_names = label_ids.copy()
    else:
        assert len(label_ids) == len(label_names)
    label_ids.insert(0, 'label_id')
    label_names.insert(0, 'label_name')

    means = ['mean']
    sems = ['sem']
    for trg_vertices in trg_vertices_list:
        trg_data = np.mean(maps[:, trg_vertices], 1)
        means.append(str(np.mean(trg_data)))
        sems.append(str(sem(trg_data)))

    line1 = ','.join(label_ids)
    line2 = ','.join(label_names)
    line3 = ','.join(means)
    line4 = ','.join(sems)
    lines = '\n'.join([line1, line2, line3, line4])
    open(output_file, 'w+').writelines(lines)


def compare(maps1, maps2, output_nifti, output_text, mask=None, label_names=None, p_thr=1.0):
    assert maps1.shape[1] == maps2.shape[1]
    vtx_num = maps1.shape[1]
    if mask is None:
        label_ids = list(map(str, range(vtx_num)))
        trg_vertices_list = [[vtx] for vtx in range(vtx_num)]
    else:
        assert mask.ndim == 1
        assert vtx_num == mask.shape[0]
        label_ids = [int(label_id) for label_id in np.unique(mask) if label_id != 0]
        trg_vertices_list = [np.where(mask == i)[0] for i in label_ids]
        label_ids = list(map(str, label_ids))
    if label_names is None:
        label_names = label_ids.copy()
    else:
        assert len(label_ids) == len(label_names)
    label_ids.insert(0, 'label_id')
    label_names.insert(0, 'label_name')

    ts = ['t']
    ps = ['p']
    compare_map = np.zeros(vtx_num)
    for trg_vertices in trg_vertices_list:
        trg_data1 = np.mean(np.atleast_2d(maps1[:, trg_vertices]), 1)
        trg_data2 = np.mean(np.atleast_2d(maps2[:, trg_vertices]), 1)
        t, p = ttest_ind(trg_data1, trg_data2)
        ts.append(str(t))
        ps.append(str(p))
        compare_map[trg_vertices] = t if p < p_thr else 0

    if p_thr == 1:
        line1 = ','.join(label_ids)
        line2 = ','.join(label_names)
        line3 = ','.join(ts)
        line4 = ','.join(ps)
        lines = '\n'.join([line1, line2, line3, line4])
        open(output_text, 'w+').writelines(lines)
    save2nifti(output_nifti, compare_map)


def plot_mean_sem(mean_sem_files, items=None, label_ids=None, xlabel='', ylabel=''):

    fig, ax = plt.subplots()
    x = None
    width = None
    xticklabels = None
    rects_list = []
    item_num = len(mean_sem_files)
    for idx, mean_sem_file in enumerate(mean_sem_files):
        mean_sem_dict = CsvReader(mean_sem_file).to_dict(1)
        if label_ids is None:
            label_ids = mean_sem_dict['label_id']
        else:
            label_ids = [str(i) for i in label_ids]
        if x is None:
            xticklabels = [mean_sem_dict['label_name'][mean_sem_dict['label_id'].index(i)] for i in label_ids]
            x = np.arange(len(label_ids))
            width = auto_bar_width(x, item_num)
        y = [float(mean_sem_dict['mean'][mean_sem_dict['label_id'].index(i)]) for i in label_ids]
        sems = [float(mean_sem_dict['sem'][mean_sem_dict['label_id'].index(i)]) for i in label_ids]
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


def plot_compare(compare_file, label_ids=None, p_thr=None):

    file_name = os.path.basename(compare_file)
    compare_dict = CsvReader(compare_file).to_dict(1)
    if label_ids is None:
        label_ids = compare_dict['label_id']
    else:
        label_ids = [str(i) for i in label_ids]

    if p_thr is not None:
        label_ids = [i for i in label_ids if float(compare_dict['p'][compare_dict['label_id'].index(i)]) < p_thr]

    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    if len(label_ids) > 0:
        x = np.arange(len(label_ids))
        width = auto_bar_width(x)
        y_t = [float(compare_dict['t'][compare_dict['label_id'].index(i)]) for i in label_ids]
        y_p = [float(compare_dict['p'][compare_dict['label_id'].index(i)]) for i in label_ids]
        rects_t = ax.bar(x, y_t, width, color='b', alpha=0.5)
        rects_p = ax_twin.bar(x, y_p, width, color='g', alpha=0.5)
        ax.legend([rects_t, rects_p], ['t', 'p'])
        ax.set_xticks(x)
        xticklabels = [compare_dict['label_name'][compare_dict['label_id'].index(i)] for i in label_ids]
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
    from statsmodels.stats.multitest import multipletests
    # https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html

    brain_structure = {
        'l': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'r': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/glasser_mmp_bak')
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

    # --------------------------------------calc_mean_sem start---------------------------------------------
    # items = ['r1_FFA1_connect_l1', 'r1_FFA1_connect_l2', 'r1_FFA1_connect_r1', 'r1_FFA1_connect_r2',
    #          'r1_FFA2_connect_l1', 'r1_FFA2_connect_l2', 'r1_FFA2_connect_r1', 'r1_FFA2_connect_r2',
    #          'r2_FFA1_connect_l2', 'r2_FFA1_connect_l1', 'r2_FFA1_connect_r2', 'r2_FFA1_connect_r1',
    #          'l1_FFA1_connect_l1', 'l1_FFA1_connect_l2', 'l1_FFA1_connect_r1', 'l1_FFA1_connect_r2',
    #          'l1_FFA2_connect_l1', 'l1_FFA2_connect_l2', 'l1_FFA2_connect_r1', 'l1_FFA2_connect_r2',
    #          'l2_FFA1_connect_l2', 'l2_FFA1_connect_l1', 'l2_FFA1_connect_r2', 'l2_FFA1_connect_r1']
    # mean_sem_dir = pjoin(connect_dir, 'mean_sem')
    # if not os.path.exists(mean_sem_dir):
    #     os.makedirs(mean_sem_dir)
    # reader = CiftiReader('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')
    # for item in items:
    #     maps = np.atleast_2d(nib.load(pjoin(connect_dir, '{}.nii.gz'.format(item))).get_data())
    #     mask = reader.get_data(brain_structure[item[-2]], True).ravel()
    #     label_ids = [int(label_id) for label_id in np.unique(mask) if label_id != 0]
    #     label_names = [reader.label_tables()[0][i].label for i in label_ids]
    #     calc_mean_sem(maps, pjoin(mean_sem_dir, item), mask, label_names)
    # --------------------------------------calc_mean_sem end---------------------------------------------

    # --------------------------------------calculate and plot compare start--------------------------------------------
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
    # interroi_item_pairs = [
    #     ['l2_FFA1_connect_l1', 'l1_FFA1_connect_l1'],
    #     ['l2_FFA1_connect_l1', 'l1_FFA2_connect_l1'],
    #     ['l2_FFA1_connect_r1', 'l1_FFA1_connect_r1'],
    #     ['l2_FFA1_connect_r1', 'l1_FFA2_connect_r1'],
    #     ['r2_FFA1_connect_l1', 'r1_FFA1_connect_l1'],
    #     ['r2_FFA1_connect_l1', 'r1_FFA2_connect_l1'],
    #     ['r2_FFA1_connect_r1', 'r1_FFA1_connect_r1'],
    #     ['r2_FFA1_connect_r1', 'r1_FFA2_connect_r1'],
    #     ['l2_FFA1_connect_l2', 'l1_FFA1_connect_l2'],
    #     ['l2_FFA1_connect_l2', 'l1_FFA2_connect_l2'],
    #     ['l2_FFA1_connect_r2', 'l1_FFA1_connect_r2'],
    #     ['l2_FFA1_connect_r2', 'l1_FFA2_connect_r2'],
    #     ['r2_FFA1_connect_l2', 'r1_FFA1_connect_l2'],
    #     ['r2_FFA1_connect_l2', 'r1_FFA2_connect_l2'],
    #     ['r2_FFA1_connect_r2', 'r1_FFA1_connect_r2'],
    #     ['r2_FFA1_connect_r2', 'r1_FFA2_connect_r2']
    # ]
    # compare_dir = pjoin(connect_dir, 'compare')
    # if not os.path.exists(compare_dir):
    #     os.makedirs(compare_dir)
    # reader = CiftiReader('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')
    # for item1, item2 in intersubgroup_item_pairs:
    #     maps1 = np.atleast_2d(nib.load(pjoin(connect_dir, '{}.nii.gz'.format(item1))).get_data())
    #     maps2 = np.atleast_2d(nib.load(pjoin(connect_dir, '{}.nii.gz'.format(item2))).get_data())
    #     mask = reader.get_data(brain_structure[item1[-2]], True).ravel()
    #     label_ids = [int(label_id) for label_id in np.unique(mask) if label_id != 0]
    #     label_names = [reader.label_tables()[0][i].label for i in label_ids]
    #     compare(maps1, maps2, pjoin(compare_dir, '{}_vs_{}.nii.gz'.format(item1, item2)),
    #             pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), mask, label_names)
    #
    # for item1, item2 in intersubgroup_item_pairs:
    #     plot_compare(pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), labels_dict[item1[-2]], 0.05)
    # --------------------------------------calculate and plot compare end--------------------------------------------

    # --------------------------------------plot_mean_sem start---------------------------------------------
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
    # mean_sem_dir = pjoin(connect_dir, 'mean_sem')
    # for items in intersubgroup_items_list:
    #     mean_sem_files = [pjoin(mean_sem_dir, item) for item in items]
    #     plot_mean_sem(mean_sem_files, items, labels_dict[items[0][-2]], ylabel='pearson r')
    # --------------------------------------plot_mean_sem end---------------------------------------------

    acti_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation')
    repre_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/representation')
    mask_files = {
        'l': pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_l.nii.gz'),
        'r': pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_r.nii.gz')
    }
    labelconfig_files = {
        'l': pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_labelconfig_l.csv'),
        'r': pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_labelconfig_r.csv')
    }

    # ---activation magnitude start---
    # items = ['l1_maps', 'l2_maps', 'r1_maps', 'r2_maps']
    # mean_sem_dir = pjoin(acti_dir, 'mean_sem_PAM_z165_p025')
    # if not os.path.exists(mean_sem_dir):
    #     os.makedirs(mean_sem_dir)
    # for item in items:
    #     maps = np.atleast_2d(nib.load(pjoin(acti_dir, '{}.nii.gz'.format(item))).get_data())
    #     mask = nib.load(mask_files[item[0]]).get_data().ravel()
    #     labelconfig = CsvReader(labelconfig_files[item[0]]).to_dict()
    #     label_names = labelconfig['label_name']
    #     calc_mean_sem(maps, pjoin(mean_sem_dir, item), mask, label_names)

    # item_pairs = [
    #     ['l1_maps', 'l2_maps'],
    #     ['r1_maps', 'r2_maps']
    # ]
    # compare_dir = pjoin(acti_dir, 'compare_PAM_z165_p025')
    # if not os.path.exists(compare_dir):
    #     os.makedirs(compare_dir)
    # for item1, item2 in item_pairs:
    #     maps1 = np.atleast_2d(nib.load(pjoin(acti_dir, '{}.nii.gz'.format(item1))).get_data())
    #     maps2 = np.atleast_2d(nib.load(pjoin(acti_dir, '{}.nii.gz'.format(item2))).get_data())
    #     mask = nib.load(mask_files[item1[0]]).get_data().ravel()
    #     labelconfig = CsvReader(labelconfig_files[item1[0]]).to_dict()
    #     label_names = labelconfig['label_name']
    #     compare(maps1, maps2, pjoin(compare_dir, '{}_vs_{}.nii.gz'.format(item1, item2)),
    #             pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), mask, label_names)

    # for item1, item2 in item_pairs:
    #     plot_compare(pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), p_thr=0.05)

    items_list = [
        ['l1_maps', 'l2_maps'],
        ['r1_maps', 'r2_maps']
    ]
    mean_sem_dir = pjoin(acti_dir, 'mean_sem_PAM_z165_p025')
    compare_dir = pjoin(acti_dir, 'compare_PAM_z165_p025')
    for items in items_list:
        compare_file = pjoin(compare_dir, '{}_vs_{}'.format(items[0], items[1]))
        compare_dict = CsvReader(compare_file).to_dict(1)
        p_uncorrected = np.array(list(map(float, compare_dict['p'])))
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_uncorrected, 0.05, 'fdr_bh')
        print('p_uncorrected:', p_uncorrected)
        print('p_corrected:', p_corrected)
        label_ids = compare_dict['label_id']
        label_ids = [i for i in label_ids if float(compare_dict['p'][compare_dict['label_id'].index(i)]) < 1]
        mean_sem_files = [pjoin(mean_sem_dir, item) for item in items]
        plot_mean_sem(mean_sem_files, items, label_ids, ylabel='activation (z-stats)')
    # ---activation magnitude end---

    # ---activation pattern start---
    # items = ['lh_intra_subgroup_dissimilarity', 'lh_inter_subgroup_dissimilarity',
    #          'rh_intra_subgroup_dissimilarity', 'rh_inter_subgroup_dissimilarity']
    # mean_sem_dir = pjoin(repre_dir, 'mean_sem_PAM_z165_p025')
    # if not os.path.exists(mean_sem_dir):
    #     os.makedirs(mean_sem_dir)
    # for item in items:
    #     maps = np.atleast_2d(nib.load(pjoin(repre_dir, '{}.nii.gz'.format(item))).get_data())
    #     mask = nib.load(mask_files[item[0]]).get_data().ravel()
    #     labelconfig = CsvReader(labelconfig_files[item[0]]).to_dict()
    #     label_names = labelconfig['label_name']
    #     calc_mean_sem(maps, pjoin(mean_sem_dir, item), mask, label_names)

    # item_pairs = [
    #     ['lh_intra_subgroup_dissimilarity', 'lh_inter_subgroup_dissimilarity'],
    #     ['rh_intra_subgroup_dissimilarity', 'rh_inter_subgroup_dissimilarity']
    # ]
    # compare_dir = pjoin(repre_dir, 'compare_PAM_z165_p025')
    # if not os.path.exists(compare_dir):
    #     os.makedirs(compare_dir)
    # for item1, item2 in item_pairs:
    #     maps1 = np.atleast_2d(nib.load(pjoin(repre_dir, '{}.nii.gz'.format(item1))).get_data())
    #     maps2 = np.atleast_2d(nib.load(pjoin(repre_dir, '{}.nii.gz'.format(item2))).get_data())
    #     mask = nib.load(mask_files[item1[0]]).get_data().ravel()
    #     labelconfig = CsvReader(labelconfig_files[item1[0]]).to_dict()
    #     label_names = labelconfig['label_name']
    #     compare(maps1, maps2, pjoin(compare_dir, '{}_vs_{}.nii.gz'.format(item1, item2)),
    #             pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), mask, label_names)

    # for item1, item2 in item_pairs:
    #     plot_compare(pjoin(compare_dir, '{}_vs_{}'.format(item1, item2)), p_thr=0.1)

    # items_list = [
    #     ['lh_intra_subgroup_dissimilarity', 'lh_inter_subgroup_dissimilarity'],
    #     ['rh_intra_subgroup_dissimilarity', 'rh_inter_subgroup_dissimilarity']
    # ]
    # mean_sem_dir = pjoin(repre_dir, 'mean_sem_PAM_z165_p025')
    # compare_dir = pjoin(repre_dir, 'compare_PAM_z165_p025')
    # for items in items_list:
    #     compare_file = pjoin(compare_dir, '{}_vs_{}'.format(items[0], items[1]))
    #     compare_dict = CsvReader(compare_file).to_dict(1)
    #     p_uncorrected = np.array(list(map(float, compare_dict['p'])))
    #     reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_uncorrected, 0.05, 'fdr_bh')
    #     print('p_uncorrected:', p_uncorrected)
    #     print('p_corrected:', p_corrected)
    #     label_ids = compare_dict['label_id']
    #     label_ids = [i for i in label_ids if p_uncorrected[compare_dict['label_id'].index(i)] < 1]
    #     mean_sem_files = [pjoin(mean_sem_dir, item) for item in items]
    #     plot_mean_sem(mean_sem_files, items, label_ids, ylabel='euclidean')
    # ---activation pattern end---
