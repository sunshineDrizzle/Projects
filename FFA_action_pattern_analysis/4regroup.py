if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.stats.stats import pearsonr
    from commontool.io.io import GiftiReader

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    lh_maps_file = pjoin(project_dir, 'data/LiuLab_face-avg/LiuLab_495_FACE-AVG_zstat_fsaverage_lh.nii.gz')
    rh_maps_file = pjoin(project_dir, 'data/LiuLab_face-avg/LiuLab_495_FACE-AVG_zstat_fsaverage_rh.nii.gz')
    lFFA_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/lFFA_25_fsaverage.label.gii')
    rFFA_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/rFFA_25_fsaverage.label.gii')
    lh_g1_mean_map_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation/lh_g1_mean_map_fsaverage.func.gii')
    lh_g2_mean_map_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation/lh_g2_mean_map_fsaverage.func.gii')
    rh_g1_mean_map_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation/rh_g1_mean_map_fsaverage.func.gii')
    rh_g2_mean_map_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation/rh_g2_mean_map_fsaverage.func.gii')
    regroup_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/regroup/LiuLab_face-avg')
    if not os.path.exists(regroup_dir):
        os.makedirs(regroup_dir)

    lh_maps = nib.load(lh_maps_file).get_data()
    rh_maps = nib.load(rh_maps_file).get_data()
    lFFA_vertices = np.where(GiftiReader(lFFA_file).scalar_data != 0)[0]
    rFFA_vertices = np.where(GiftiReader(rFFA_file).scalar_data != 0)[0]
    lh_g1_mean_map = GiftiReader(lh_g1_mean_map_file).scalar_data
    lh_g2_mean_map = GiftiReader(lh_g2_mean_map_file).scalar_data
    rh_g1_mean_map = GiftiReader(rh_g1_mean_map_file).scalar_data
    rh_g2_mean_map = GiftiReader(rh_g2_mean_map_file).scalar_data

    FFA_maps = np.c_[lh_maps[:, lFFA_vertices], rh_maps[:, rFFA_vertices]]
    g1_FFA_mean_map = np.r_[lh_g1_mean_map[lFFA_vertices], rh_g1_mean_map[rFFA_vertices]]
    g2_FFA_mean_map = np.r_[lh_g2_mean_map[lFFA_vertices], rh_g2_mean_map[rFFA_vertices]]

    regroup_labels = []
    for FFA_map in FFA_maps:
        corr_g1 = pearsonr(FFA_map, g1_FFA_mean_map)[0]
        corr_g2 = pearsonr(FFA_map, g2_FFA_mean_map)[0]
        if corr_g1 > corr_g2:
            regroup_labels.append('1')
        elif corr_g1 < corr_g2:
            regroup_labels.append('2')
        else:
            regroup_labels.append('3')
    open(pjoin(regroup_dir, 'regroup_labels'), 'w+').write(' '.join(regroup_labels))
