from commontool.io.io import CsvReader
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


def roi_mean_plot(roi_mean_file, rois, colors):
    roi_mean_dict = CsvReader(roi_mean_file).to_dict(axis=1)
    roi_means_list = [list(map(float, roi_mean_dict[roi])) for roi in rois]

    roi_num = len(rois)
    for i in range(roi_num):
        for j in range(i+1, roi_num):
            print('{} vs. {}'.format(rois[i], rois[j]),
                  ttest_ind(roi_means_list[i], roi_means_list[j]))

    plt.figure()
    violin_parts = plt.violinplot(roi_means_list, showmeans=True)
    for idx, pc in enumerate(violin_parts['bodies']):
        # https://stackoverflow.com/questions/26291479/changing-the-color-of-matplotlibs-violin-plots
        pc.set_color(colors[idx])
    plt.ylabel('face-avg')
    plt.xticks(range(1, len(rois)+1), rois)
    plt.tight_layout()


if __name__ == '__main__':
    from os.path import join as pjoin

    rois = [
        'l1_FFA1',
        'l2_FFA1',
        'l1_FFA2',
        'r1_FFA1',
        'r2_FFA1',
        'r1_FFA2'
    ]
    colors = ['blue', 'red', 'blue', 'blue', 'red', 'blue']

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    roi_dir = pjoin(project_dir, '2mm_25_HAC_ward_euclidean_zscore/2clusters/roi_analysis')

    roi_mean_plot(pjoin(roi_dir, 'roi_mean_face-avg'), rois, colors)

    plt.show()
