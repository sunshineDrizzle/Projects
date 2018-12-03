if __name__ == '__main__':
    import numpy as np

    from matplotlib import pyplot as plt
    from commontool.io.io import CsvReader

    stat_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/data/' \
                'HCP_face-avg/s2/patches_15/LV_weighted/lFFA_patch_stats_thr2.3'
    reader = CsvReader(stat_file)
    patch_nums = np.array(reader.cols[1], dtype=np.uint16)
    patch_sizes = []
    for row in reader.rows:
        patch_sizes.extend(row[2:])
    patch_sizes = np.array(patch_sizes, dtype=np.uint16)

    plt.hist(patch_nums, np.arange(1, patch_nums.max()+2), align='left', facecolor='white', edgecolor='black')
    plt.xlabel('#patch')
    plt.ylabel('count')
    plt.figure()
    plt.hist(patch_sizes, np.arange(1, patch_sizes.max()+2), align='left', facecolor='white', edgecolor='black')
    plt.xlabel('patch size / #vertices')
    plt.ylabel('count')
    plt.show()
