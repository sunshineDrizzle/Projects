if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, GiftiReader, save2nifti
    from commontool.algorithm.graph import connectivity_grow
    from commontool.algorithm.triangular_mesh import get_n_ring_neighbor

    threshold = 2.3
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    lFFA_label_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm_15.label')
    rFFA_label_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm_15.label')
    maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    subject_ids_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15')
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    lFFA_vertices = nib.freesurfer.read_label(lFFA_label_file)
    lFFA_vertices = set(lFFA_vertices.astype(np.uint16))
    rFFA_vertices = nib.freesurfer.read_label(rFFA_label_file)
    rFFA_vertices = set(rFFA_vertices.astype(np.uint16))
    c_reader = CiftiReader(maps_file)
    lmaps = c_reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
    rmaps = c_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    with open(subject_ids_file) as rf:
        subject_ids = rf.read().splitlines()

    g_reader_l = GiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
                             'HCP_S1200_GroupAvg_v1/S1200.L.white_MSMAll.32k_fs_LR.surf.gii')
    g_reader_r = GiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
                             'HCP_S1200_GroupAvg_v1/S1200.R.white_MSMAll.32k_fs_LR.surf.gii')

    lFFA_patch_maps = np.zeros_like(lmaps)
    lFFA_patch_stats = []
    for idx, lmap in enumerate(lmaps):
        patch_stat = [subject_ids[idx]]

        vertices_thr = np.where(lmap > threshold)[0]
        vertices_thr_FFA = lFFA_vertices.intersection(vertices_thr)
        mask = np.zeros(g_reader_l.coords.shape[0])
        mask[list(vertices_thr_FFA)] = 1
        edge_list = get_n_ring_neighbor(g_reader_l.faces, mask=mask)
        patch_num = 0
        patch_sizes = []
        while vertices_thr_FFA:
            patch_num += 1
            seed = vertices_thr_FFA.pop()
            patch = connectivity_grow([[seed]], edge_list)[0]
            patch_sizes.append(str(len(patch)))
            lFFA_patch_maps[idx, list(patch)] = patch_num
            vertices_thr_FFA.difference_update(patch)
        patch_stat.append(str(patch_num))
        patch_stat.extend(patch_sizes)
        lFFA_patch_stats.append(','.join(patch_stat))

    rFFA_patch_maps = np.zeros_like(rmaps)
    rFFA_patch_stats = []
    for idx, rmap in enumerate(rmaps):
        patch_stat = [subject_ids[idx]]

        vertices_thr = np.where(rmap > threshold)[0]
        vertices_thr_FFA = rFFA_vertices.intersection(vertices_thr)
        mask = np.zeros(g_reader_r.coords.shape[0])
        mask[list(vertices_thr_FFA)] = 1
        edge_list = get_n_ring_neighbor(g_reader_r.faces, mask=mask)
        patch_num = 0
        patch_sizes = []
        while vertices_thr_FFA:
            patch_num += 1
            seed = vertices_thr_FFA.pop()
            patch = connectivity_grow([[seed]], edge_list)[0]
            patch_sizes.append(str(len(patch)))
            rFFA_patch_maps[idx, list(patch)] = patch_num
            vertices_thr_FFA.difference_update(patch)
        patch_stat.append(str(patch_num))
        patch_stat.extend(patch_sizes)
        rFFA_patch_stats.append(','.join(patch_stat))

    save2nifti(pjoin(patch_dir, 'lFFA_patch_maps_thr{}.nii.gz'.format(threshold)), lFFA_patch_maps)
    save2nifti(pjoin(patch_dir, 'rFFA_patch_maps_thr{}.nii.gz'.format(threshold)), rFFA_patch_maps)
    open(pjoin(patch_dir, 'lFFA_patch_stats_thr{}'.format(threshold)), 'w+').writelines('\n'.join(lFFA_patch_stats))
    open(pjoin(patch_dir, 'rFFA_patch_stats_thr{}'.format(threshold)), 'w+').writelines('\n'.join(rFFA_patch_stats))
