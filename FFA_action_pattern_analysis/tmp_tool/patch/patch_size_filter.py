if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import save2nifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/LV_unweighted')
    patch_file = pjoin(patch_dir, 'rFFA_patch_maps_thr2.3.nii.gz')
    patch_file_filtered = pjoin(patch_dir, 'rFFA_patch_maps_thr2.3_lt5.nii.gz')

    patch_maps = nib.load(patch_file).get_data()
    patch_maps_filtered = np.zeros_like(patch_maps)
    label_new = 0
    for row in range(patch_maps.shape[0]):
        labels = np.unique(patch_maps[row])
        for label in labels:
            if label == 0:
                continue
            vertices = np.where(patch_maps[row] == label)[0]
            if len(vertices) > 5:
                label_new += 1
                patch_maps_filtered[row, vertices] = label_new

    header = nib.Nifti2Header()
    header['descrip'] = 'FreeROI label'
    save2nifti(patch_file_filtered, patch_maps_filtered, header=header)
