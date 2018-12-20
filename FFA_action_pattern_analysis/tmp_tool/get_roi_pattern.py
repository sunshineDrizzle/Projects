import numpy as np

from scipy import stats
from commontool.algorithm.triangular_mesh import get_n_ring_neighbor
from FFA_action_pattern_analysis.tmp_tool.patch.get_patches import get_patch_by_crg


def get_roi_pattern(maps, roi, zscore=False, thr=None, bin=False, size_min=0, faces=None, mask=None):
    """

    :param maps: N x M array
        N subjects' hemisphere map
    :param roi: list|1D array
        a collection of vertices of the ROI
    :param zscore: bool
        If True, do z-score on each subject's ROI pattern.
        It will be ignored when 'bin' is True.
    :param thr: float
        A threshold used to cut ROI data before clustering (default: None)
    :param bin: bool
        If True, binarize ROI data according to 'thr'.
        It will be ignored when 'thr' is None.
    :param size_min: non-negative integer
        If is less than or equal to 0, do nothing.
        else, only reserve the patches whose size is larger than 'size_min' after threshold. And
        'faces' must be provided.
        It will be ignored when 'thr' is None or 'mask' is not None.
    :param faces: face_num x 3 array
        It only takes effect when 'size_min' is working.
    :param mask: N x M array
        indices array used to specify valid vertices
        It will be ignored when 'thr' is None

    :return: patterns: N x len(roi) array
        N subjects' ROI pattern
    """
    tmp_maps = maps.copy()
    if thr is not None:
        if mask is not None:
            tmp_maps[np.logical_not(mask)] = thr
        elif size_min > 0:
            patch_maps = np.zeros_like(maps, dtype=np.bool)
            for row in range(patch_maps.shape[0]):
                vertices_thr = set(np.where(maps[row] > thr)[0])
                vertices_thr_roi = vertices_thr.intersection(roi)
                mask = np.zeros(patch_maps.shape[1])
                mask[list(vertices_thr_roi)] = 1
                edge_list = get_n_ring_neighbor(faces, mask=mask)
                patches = get_patch_by_crg(vertices_thr_roi, edge_list)
                for patch in patches:
                    if len(patch) > size_min:
                        patch_maps[row, patch] = True
            tmp_maps[np.logical_not(patch_maps)] = thr
        patterns = tmp_maps[:, roi]
        if bin:
            patterns = (patterns > thr).astype(np.int8)
        else:
            patterns[patterns <= thr] = thr
    else:
        patterns = tmp_maps[:, roi]

    if zscore and not bin:
        patterns = stats.zscore(patterns, 1)

    return patterns


if __name__ == '__main__':
    import os
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, GiftiReader, save2nifti

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    zscore = True  # If true, do z-score on each subject's FFA pattern
    thr = None  # a threshold used to cut FFA_data before clustering (default: None)
    bin = False  # If true, binarize FFA_data according to clustering_thr
    size_min = 0  # only work with threshold

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    analysis_dir = pjoin(project_dir, 's2_25_zscore')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    FFA_label_files = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label')
    maps_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    geo_files = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
                'HCP_S1200_GroupAvg_v1/S1200.{}.white_MSMAll.32k_fs_LR.surf.gii'
    # mask_file = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/crg2.3/{}FFA_patch_maps_lt15.nii.gz')
    mask_file = None
    # -----------------------
    print('Finish: predefine some variates')

    reader = CiftiReader(maps_file)
    if mask_file is not None:
        lh_mask = nib.load(mask_file.format('l')).get_data() != 0
        rh_mask = nib.load(mask_file.format('r')).get_data() != 0
    else:
        lh_mask = None
        rh_mask = None
    lh_geo_reader = GiftiReader(geo_files.format('L'))
    rh_geo_reader = GiftiReader(geo_files.format('R'))
    lFFA_vertices = nib.freesurfer.read_label(FFA_label_files.format('l'))
    rFFA_vertices = nib.freesurfer.read_label(FFA_label_files.format('r'))
    lFFA_patterns = get_roi_pattern(reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True), lFFA_vertices,
                                    zscore, thr, bin, size_min, lh_geo_reader.faces, lh_mask)
    rFFA_patterns = get_roi_pattern(reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True), rFFA_vertices,
                                    zscore, thr, bin, size_min, rh_geo_reader.faces, rh_mask)

    save2nifti(pjoin(analysis_dir, 'lFFA_patterns.nii.gz'), lFFA_patterns)
    save2nifti(pjoin(analysis_dir, 'rFFA_patterns.nii.gz'), rFFA_patterns)
