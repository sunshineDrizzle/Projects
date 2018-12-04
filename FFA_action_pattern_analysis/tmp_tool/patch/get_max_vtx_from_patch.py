if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    acti_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/LV_unweighted')
    patch_file = pjoin(patch_dir, 'rFFA_patch_maps_thr2.3.nii.gz')
    max_maps_file = pjoin(patch_dir, 'rFFA_max_maps_thr2.3.nii.gz')
    all_max_map_file = pjoin(patch_dir, 'rFFA_all_max_map_thr2.3.nii.gz')
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'

    acti_maps = CiftiReader(acti_file).get_data(brain_structure, True)
    patch_maps = nib.load(patch_file).get_data()
    max_maps = np.zeros_like(patch_maps)
    labels = np.unique(patch_maps)
    for label in labels:
        acti_maps_tmp = acti_maps.copy()
        not_label_indices = np.logical_not(patch_maps == label)
        acti_maps_tmp[not_label_indices] = -np.inf
        max_indices = np.argmax(acti_maps_tmp, 1)
        max_maps[range(len(max_indices)), max_indices] = label
    all_max_map = (np.sum(max_maps, 0) > 0).astype(dtype=np.uint8)

    save2nifti(max_maps_file, max_maps)
    save2nifti(all_max_map_file, all_max_map)
