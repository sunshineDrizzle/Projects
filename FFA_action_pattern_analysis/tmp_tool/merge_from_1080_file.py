if __name__ == '__main__':
    import os
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    items = [
        'activation',
        'curvature'
    ]
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    subject_ids_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    subject_ids_selected_file = pjoin(project_dir, '2mm_KM_thr2.3_zscore/3clusters/random_from_subgroup/subject3_id_random10')
    out_dir = os.path.dirname(subject_ids_selected_file)
    out_file = pjoin(out_dir, '{hemi}_{item}3_random10.nii.gz')

    with open(subject_ids_file) as rf:
        subject_ids = rf.read().splitlines()
    with open(subject_ids_selected_file) as rf:
        subject_ids_selected = rf.read().splitlines()

    for item in items:
        if item == 'activation':
            maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
        elif item == 'curvature':
            maps_file = pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii')
        else:
            raise RuntimeError("{} is not supported at present!".format(item))

        reader = CiftiReader(maps_file)
        lmaps = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
        rmaps = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)

        lmaps_selected = []
        rmaps_selected = []
        for subject_id in subject_ids_selected:
            subject_idx = subject_ids.index(subject_id)
            lmaps_selected.append(lmaps[subject_idx])
            rmaps_selected.append(rmaps[subject_idx])
        save2nifti(out_file.format(hemi='lh', item=item), np.array(lmaps_selected))
        save2nifti(out_file.format(hemi='rh', item=item), np.array(rmaps_selected))
