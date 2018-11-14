if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine some variates
    n_clusters = 10
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, '2mm_ward_regress')
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))

    # curv_reader = CiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
    #                           'HCP_S1200_GroupAvg_v1/S1200.All.curvature_MSMAll.32k_fs_LR.dscalar.nii')
    curv_reader = CiftiReader(pjoin(project_dir, 'data/1080_curvature.dscalar.nii'))
    curv_data = curv_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    sulc_label = curv_data < 0
    curv_names = curv_reader.map_names()

    curv_mean_maps = np.zeros((0, curv_data.shape[1]))
    sulc_prob_maps = np.zeros((0, curv_data.shape[1]))
    for label in range(1, n_clusters+1):
        with open(pjoin(n_clusters_dir, 'subjects{}_id'.format(label))) as f:
            subgroup_ids = f.read().splitlines()
        subgroup_curv_names = [name + '_Curvature' for name in subgroup_ids]

        indices = []
        for idx, name in enumerate(curv_names):
            if name in subgroup_curv_names:
                indices.append(idx)

        subgroup_curv_data = curv_data[indices, :]
        subgroup_curv_data_mean = np.atleast_2d(np.mean(subgroup_curv_data, 0))
        subgroup_sulc_prob = np.atleast_2d(np.mean(sulc_label[indices, :], 0))

        curv_mean_maps = np.r_[curv_mean_maps, subgroup_curv_data_mean]
        sulc_prob_maps = np.r_[sulc_prob_maps, subgroup_sulc_prob]

        print(label, subgroup_curv_data.shape[0])

    # output
    save2nifti(pjoin(n_clusters_dir, 'curv_mean_maps_new.nii.gz'), curv_mean_maps)
    # save2nifti(pjoin(n_clusters_dir, 'sulc_prob_maps.nii.gz'), sulc_prob_maps)
