if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import save2nifti

    size_min = 15
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/crg')
    patch_file = pjoin(patch_dir, 'rFFA_patch_maps.nii.gz')
    patch_file_filtered = pjoin(patch_dir, 'rFFA_patch_maps_lt15.nii.gz')
    stat_file = pjoin(patch_dir, 'rFFA_patch_stats_lt15')
    subject_ids_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    with open(subject_ids_file) as rf:
        subject_ids = rf.read().splitlines()

    patch_maps = nib.load(patch_file).get_data()
    patch_maps_filtered = np.zeros_like(patch_maps)
    patch_stats = []
    label_new = 0
    for row in range(patch_maps.shape[0]):
        labels = np.unique(patch_maps[row])
        patch_stat = [subject_ids[row]]
        patch_sizes = []
        for label in labels:
            if label == 0:
                continue
            vertices = np.where(patch_maps[row] == label)[0]
            size = len(vertices)
            if size > size_min:
                label_new += 1
                patch_sizes.append(str(size))
                patch_maps_filtered[row, vertices] = label_new
        patch_stat.append(str(label_new))
        patch_stat.extend(patch_sizes)
        patch_stats.append(','.join(patch_stat))
        label_new = 0

    header = nib.Nifti2Header()
    header['descrip'] = 'FreeROI label'
    save2nifti(patch_file_filtered, patch_maps_filtered, header=header)
    open(stat_file, 'w+').writelines('\n'.join(patch_stats))
