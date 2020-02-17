

def rois_stats():
    import numpy as np
    import nibabel as nib
    import pickle as pkl

    from os.path import join as pjoin
    from commontool.io.io import save2nifti

    roi_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/s2_rh/zscore/HAC_ward_euclidean/100clusters/activation/ROIs/v3'
    roi_file = pjoin(roi_dir, 'rois.nii.gz')
    roi2label_file = pjoin(roi_dir, 'roi2label.csv')
    group_label_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/' \
                       'analysis/s2_rh/zscore/HAC_ward_euclidean/100clusters/group_labels'

    rois = nib.load(roi_file).get_data().squeeze().T
    group_labels_all = np.array(open(group_label_file).read().split(' '), dtype=np.int)
    roi2label = dict()
    for line in open(roi2label_file).read().splitlines():
        k, v = line.split(',')
        roi2label[k] = int(v)

    # prepare rois information dict
    rois_info = dict()
    for roi in roi2label.keys():
        rois_info[roi] = dict()

    prob_maps = []
    for roi, label in roi2label.items():
        # get indices of groups which contain the roi
        indices = rois == label
        group_indices = np.any(indices, 1)

        # calculate the number of the valid groups
        n_group = np.sum(group_indices)
        rois_info[roi]['n_group'] = n_group

        # calculate the number of the subjects
        n_subj = 0
        group_labels = np.where(group_indices)[0] + 1
        for i in group_labels:
            n_subj += np.sum(group_labels_all == i)
        rois_info[roi]['n_subject'] = n_subj

        # calculate roi sizes for each valid group
        sizes = np.sum(indices[group_indices], 1)
        rois_info[roi]['sizes'] = sizes

        # calculate roi probability map among valid groups
        prob_map = np.mean(indices[group_indices], 0)
        prob_maps.append(prob_map)
    prob_maps = np.array(prob_maps).T[:, None, None, :]

    # save out
    pkl.dump(rois_info, open(pjoin(roi_dir, 'rois_info.pkl'), 'wb'))
    save2nifti(pjoin(roi_dir, 'prob_maps.nii.gz'), prob_maps)


def plot_roi_info():
    import numpy as np
    import pickle as pkl

    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import show_bar_value, auto_bar_width

    roi_info_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/s2_rh/zscore/' \
                    'HAC_ward_euclidean/100clusters/activation/ROIs/v3/rois_info.pkl'
    roi_infos = pkl.load(open(roi_info_file, 'rb'))

    # -plot n_group and n_subject-
    x_labels = list(roi_infos.keys())
    n_roi = len(x_labels)
    x = np.arange(n_roi)
    width = auto_bar_width(x)

    # plot n_group
    y_n_group = [info['n_group'] for info in roi_infos.values()]
    rects_group = plt.bar(x, y_n_group, width, facecolor='white', edgecolor='black')
    show_bar_value(rects_group)
    plt.xticks(x, x_labels)
    plt.ylabel('#group')

    # plot n_subject
    plt.figure()
    y_n_subj = [info['n_subject'] for info in roi_infos.values()]
    rects_subj = plt.bar(x, y_n_subj, width, facecolor='white', edgecolor='black')
    show_bar_value(rects_subj)
    plt.xticks(x, x_labels)
    plt.ylabel('#subject')

    # -plot sizes-
    for roi, info in roi_infos.items():
        plt.figure()
        sizes = info['sizes']
        bins = np.linspace(min(sizes), max(sizes), 50)
        _, _, patches = plt.hist(sizes, bins, color='white', edgecolor='black')
        plt.xlabel('#vertex')
        plt.title(f'distribution of {roi} sizes')
        show_bar_value(patches, '.0f')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # rois_stats()
    plot_roi_info()
