import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy import stats


def get_scalars_on_PA(src_file, PA, axes, color, label, zscore):
    maps = nib.load(src_file).get_data()

    x = np.arange(len(PA))
    ys = maps[:, PA]
    if zscore:
        ys = stats.zscore(ys, 1)
    for idx, y in enumerate(ys):
        axes[idx].plot(x, y, color=color, label=label)
        axes[idx].set_title(idx+1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cluster_num = 3
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, '2mm_KM_zscore/{}clusters'.format(cluster_num))
    lh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm_PA.label')
    rh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm_PA.label')

    lh_PA = nib.freesurfer.read_label(lh_PA_file)[-1::-1]  # from posterior to anterior
    rh_PA = nib.freesurfer.read_label(rh_PA_file)[-1::-1]  # from posterior to anterior

    fig, axes = plt.subplots(cluster_num, 2, sharex=True)
    axes[0, 0].set_title('lh')
    axes[0, 1].set_title('rh')
    axes[-1, 0].set_xlabel('from posterior to anterior of lFFA')
    axes[-1, 1].set_xlabel('from posterior to anterior of rFFA')

    get_scalars_on_PA(pjoin(cluster_num_dir, 'activation/lh_zscore_mean_maps.nii.gz'),
                      lh_PA, axes[:, 0], 'k', 'pattern', False)
    get_scalars_on_PA(pjoin(cluster_num_dir, 'activation/rh_zscore_mean_maps.nii.gz'),
                      rh_PA, axes[:, 1], 'k', 'pattern', False)

    # get_scalars_on_PA(pjoin(cluster_num_dir, 'activation/lh_mean_maps.nii.gz'),
    #                   lh_PA, axes[:, 0], 'y', 'activation', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'activation/rh_mean_maps.nii.gz'),
    #                   rh_PA, axes[:, 1], 'y', 'activation', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/lh_curv_mean_maps.nii.gz'),
    #                   lh_PA, axes[:, 0], 'r', 'curvature', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/rh_curv_mean_maps.nii.gz'),
    #                   rh_PA, axes[:, 1], 'r', 'curvature', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/lh_thickness_mean_maps.nii.gz'),
    #                   lh_PA, axes[:, 0], 'g', 'thickness', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/rh_thickness_mean_maps.nii.gz'),
    #                   rh_PA, axes[:, 1], 'g', 'thickness', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/lh_myelin_mean_maps.nii.gz'),
    #                   lh_PA, axes[:, 0], 'b', 'myelin', True)
    # get_scalars_on_PA(pjoin(cluster_num_dir, 'structure/rh_myelin_mean_maps.nii.gz'),
    #                   rh_PA, axes[:, 1], 'b', 'myelin', True)

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            axes[row, col].set_title(row+1)
            # axes[row, col].legend()

    plt.tight_layout()
    plt.show()
