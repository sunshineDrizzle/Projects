if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.io.io import CsvReader

    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = 3
    subproject_name = '2mm_KM_init10_regress_right'
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    roi2color = {'r1_FFA_m': 'b',
                 'r2_FFA_p': 'r',
                 'r2_FFA_a': 'g',
                 'r3_FFA_p': 'y',
                 'r3_FFA_a': 'k'}
    color2facecolor = {'b': 'blue',
                       'r': 'red',
                       'g': 'green',
                       'y': 'yellow',
                       'k': 'black'}

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/FFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))
    fingerprint_files = pjoin(n_clusters_dir, '{}_func_fingerprint.csv')
    # -----------------------

    fig, ax = plt.subplots()
    is_first_loop = True
    for roi, color in roi2color.items():
        fingerprint_file = fingerprint_files.format(roi)
        reader = CsvReader(fingerprint_file)
        fingerprints = np.array(reader.rows[1:], dtype=np.float64)

        x = np.arange(fingerprints.shape[1])
        fingerprints_mean = np.mean(fingerprints, axis=0)
        fingerprints_std = np.std(fingerprints, axis=0)
        ax.plot(x, fingerprints_mean, '{}.-'.format(color), label=roi)
        # ax.fill_between(x, fingerprints_mean - fingerprints_std, fingerprints_mean + fingerprints_std,
        #                 alpha=0.5, facecolors=color2facecolor[color])
        ax.legend()

        if is_first_loop:
            is_first_loop = False
            plt.xticks(x, reader.rows[0])

    plt.show()
