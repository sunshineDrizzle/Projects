def get_roi_pattern():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from FFA_pattern.tool import get_roi_pattern

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'rh'
    zscore = True  # If true, do z-score on each subject's FFA pattern
    thr = None  # a threshold used to cut FFA_data before clustering (default: None)
    bin = False  # If true, binarize FFA_data according to clustering_thr
    size_min = 0  # only work with threshold

    # predefine paths
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    trg_dir = pjoin(proj_dir, f'analysis/clustering_{hemi}/zscore')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)
    roi_file = pjoin(proj_dir, f'data/HCP/label/MMPprob_OFA_FFA_thr1_{hemi}.label')
    activ_file = pjoin(trg_dir, 'activation.nii.gz')
    # geo_files = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
    #             'HCP_S1200_GroupAvg_v1/S1200.{}.white_MSMAll.32k_fs_LR.surf.gii'
    geo_files = None
    # mask_file = pjoin(proj_dir, f'data/HCP_face-avg/s2/patches_15/crg2.3/{hemi[0]}FFA_patch_maps_lt15.nii.gz')
    mask_file = None
    # -----------------------
    print('Finish: predefine some variates')

    if mask_file is not None:
        mask = nib.load(mask_file).get_data().squeeze().T != 0
    else:
        mask = None
    if geo_files is None:
        faces = None
    else:
        raise NotImplementedError

    activ = nib.load(activ_file).get_data().squeeze().T
    roi = nib.freesurfer.read_label(roi_file)
    roi_patterns = get_roi_pattern(activ, roi, zscore, thr, bin, size_min, faces, mask)

    np.save(pjoin(trg_dir, 'roi_pattern.npy'), roi_patterns)


def clustering():
    import os
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import imshow
    from commontool.algorithm.cluster import hac_scipy
    from FFA_pattern.tool import k_means, louvain_community, girvan_newman_community

    print('Start: predefine some variates')
    # -----------------------
    # predefine parameters
    hemi = 'rh'  # 'lh', 'rh', 'both'
    clustering_method = 'HAC_ward_euclidean'  # 'HAC_average_dice', 'KM', 'LV', 'GN'
    max_n_cluster = 100
    is_graph_needed = True if clustering_method in ('LV', 'GN') else False
    weight_type = ('dissimilar', 'euclidean')  # only work when is_graph_needed is True

    # predefine paths
    src_dir = f'/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/clustering_{hemi}/zscore'
    trg_dir = pjoin(src_dir, f'{clustering_method}/results')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    roi_pattern_file = pjoin(src_dir, 'roi_pattern.npy')
    # -----------------------
    print('Finish: predefine some variates')

    print('Start: prepare data')
    # -----------------------
    # prepare patterns
    roi_patterns = np.load(roi_pattern_file)

    # show patterns
    imshow(roi_patterns, 'vertices', 'subjects', 'jet', 'pattern')
    # -----------------------
    print('Finish: prepare data')

    # structure graph
    # -----------------------
    if is_graph_needed:
        from commontool.algorithm.graph import array2graph

        print('Start: structure graph')
        graph = array2graph(roi_patterns, weight_type, edges='upper_right_triangle')
        print('Finish: structure graph')
    else:
        graph = None
    # -----------------------

    print('Start: do clustering')
    # -----------------------
    if clustering_method == 'LV':
        labels_list = louvain_community(graph)
    elif clustering_method == 'GN':
        labels_list = girvan_newman_community(graph, max_n_cluster)
    elif 'HAC' in clustering_method:
        values = clustering_method.split('_')
        labels_list = hac_scipy(roi_patterns, range(1, max_n_cluster+1), method=values[1], metric=values[2],
                                out_path=pjoin(trg_dir, 'hac_dendrogram.png'))
    elif clustering_method == 'KM':
        labels_list = k_means(roi_patterns, range(1, max_n_cluster+1), 10)
    else:
        raise RuntimeError('The clustering_method-{} is not supported!'.format(clustering_method))

    # output labels
    for labels in labels_list:
        n_label = len(set(labels))
        labels_out = ' '.join([str(label) for label in labels])
        with open(pjoin(trg_dir, '{}group_labels'.format(n_label)), 'w') as wf:
            wf.write(labels_out)
    # -----------------------
    print('Finish: do clustering')

    plt.show()


if __name__ == '__main__':
    # get_roi_pattern()
    clustering()
