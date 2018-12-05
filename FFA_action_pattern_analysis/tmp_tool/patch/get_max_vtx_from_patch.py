if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    acti_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/LV_unweighted')
    patch_file = pjoin(patch_dir, 'lFFA_patch_maps_thr2.3_lt5.nii.gz')
    max_maps_file = pjoin(patch_dir, 'lFFA_max_maps_thr2.3_lt5.nii.gz')
    prob_max_map_file = pjoin(patch_dir, 'lFFA_prob_max_map_thr2.3_lt5.nii.gz')
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'

    acti_maps = CiftiReader(acti_file).get_data(brain_structure, True)
    patch_maps = nib.load(patch_file).get_data()
    max_maps = np.zeros_like(patch_maps)
    for row in range(acti_maps.shape[0]):
        labels = np.unique(patch_maps[row])
        for label in labels:
            if label == 0:
                continue
            acti_map_tmp = acti_maps[row].copy()
            not_label_indices = np.logical_not(patch_maps[row] == label)
            acti_map_tmp[not_label_indices] = -np.inf
            max_idx = np.argmax(acti_map_tmp)
            max_maps[row, max_idx] = label
    prob_max_map = np.mean(max_maps > 0, 0)

    header = nib.Nifti2Header()
    header['descrip'] = 'FreeROI label'
    save2nifti(max_maps_file, max_maps, header=header)
    save2nifti(prob_max_map_file, prob_max_map)
