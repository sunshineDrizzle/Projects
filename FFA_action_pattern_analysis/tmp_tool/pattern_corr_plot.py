

if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from scipy.stats.stats import ttest_ind

    item1 = 'acti_corr_MBS_rFFA_1080'
    item2 = 'g1_MBS_corr_g2_MBS_rFFA'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    corr_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/pattern_corr')
    corr_file1 = pjoin(corr_dir, '{}.npy'.format(item1))
    corr_file2 = pjoin(corr_dir, '{}.npy'.format(item2))

    corr_arr1 = np.load(corr_file1)
    corr_arr2 = np.load(corr_file2)
    print(ttest_ind(corr_arr1[:, 0], corr_arr2[:, 0]))
    plt.violinplot([corr_arr1[:, 0], corr_arr2[:, 0]], showmeans=True)
    plt.ylabel('r')
    plt.xticks([1, 2], [item1, item2])

    plt.show()
