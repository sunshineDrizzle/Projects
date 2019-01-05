import os
import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy.stats import ttest_ind
from commontool.io.io import save2nifti, CiftiReader


def calc_mean(connect_files, items):
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


def compare(connect_files, item_pairs, roi_map_file=None, p_thr=1.0):
    trg_vertices_dict_lr = dict()
    if roi_map_file is not None:
        reader = CiftiReader(roi_map_file)
        roi_map_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
        roi_map_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
        trg_vertices_list_l = [np.where(roi_map_l == i)[1] for i in np.unique(roi_map_l) if i != 0]
        trg_vertices_list_r = [np.where(roi_map_r == i)[1] for i in np.unique(roi_map_r) if i != 0]
        trg_vertices_dict_lr['l'] = trg_vertices_list_l
        trg_vertices_dict_lr['r'] = trg_vertices_list_r

    connect_dir = os.path.dirname(connect_files)
    compare_dir = pjoin(connect_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    for item0, item1 in item_pairs:
        assert item0[-2] == item1[-2]
        data0 = np.atleast_2d(nib.load(connect_files.format(item=item0)).get_data())
        data1 = np.atleast_2d(nib.load(connect_files.format(item=item1)).get_data())
        assert data0.shape[1] == data1.shape[1]

        if trg_vertices_dict_lr:
            trg_vertices_list = trg_vertices_dict_lr[item0[-2]]
        else:
            trg_vertices_list = [[vtx] for vtx in range(data0.shape[1])]
        compares = np.zeros(data0.shape[1])
        for trg_vertices in trg_vertices_list:
            trg_data0 = data0[:, trg_vertices]
            trg_data1 = data1[:, trg_vertices]
            t, p = ttest_ind(np.mean(trg_data0, 1), np.mean(trg_data1, 1))
            compares[trg_vertices] = t if p < p_thr else 0

        if p_thr == 1:
            out_name = pjoin(compare_dir, '{}_vs_{}.nii.gz'.format(item0, item1))
        else:
            out_name = pjoin(compare_dir, '{}_vs_{}_p{}.nii.gz'.format(item0, item1, p_thr))
        save2nifti(out_name, compares)


if __name__ == '__main__':

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/glasser_mmp')
    connect_files = pjoin(connect_dir, '{item}.nii.gz')

    # items = ['r1_FFA1_connect_l1', 'r1_FFA1_connect_r1',
    #          'r2_FFA1_connect_l2', 'r2_FFA1_connect_r2',
    #          'r1_FFA2_connect_l1', 'r1_FFA2_connect_r1']
    # calc_mean(connect_files, items)

    item_pairs = [
        ['r2_FFA1_connect_l2', 'r2_FFA1_connect_l1'],
        ['r2_FFA1_connect_r2', 'r2_FFA1_connect_r1'],
        ['r1_FFA1_connect_l1', 'r1_FFA1_connect_l2'],
        ['r1_FFA1_connect_r1', 'r1_FFA1_connect_r2'],
        ['r1_FFA2_connect_l1', 'r1_FFA2_connect_l2'],
        ['r1_FFA2_connect_r1', 'r1_FFA2_connect_r2']
    ]
    compare(connect_files, item_pairs,
            '/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii', 0.05)
