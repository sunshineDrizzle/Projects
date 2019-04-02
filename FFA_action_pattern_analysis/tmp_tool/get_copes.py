if __name__ == '__main__':
    import numpy as np

    from collections import OrderedDict
    from commontool.io.io import CiftiReader, save2cifti

    interested_copes = OrderedDict([
        (19, 'FACE-AVG')
    ])
    subject_id_file = '/nfs/t3/workingshop/chenxiayu/data/HCP/S1200_WM_all_cope_s2_retest/subject_id'
    src_files = '/nfs/t3/workingshop/chenxiayu/data/HCP/S1200_WM_all_cope_s2_retest/' \
                '{}_tfMRI_WM_level2_hp200_s2_MSMAll.dscalar.nii'
    trg_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_clustering/data/HCP_1080/face-avg_s2/retest/' \
               'S1200_retest_WM_cope{0}_{1}_s2_MSMAll_32k_fs_LR.dscalar.nii'

    with open(subject_id_file) as rf:
        subject_ids = rf.read().splitlines()

    maps_new_dict = OrderedDict()
    map_names_new_dict = OrderedDict()
    for k in interested_copes.keys():
        maps_new_dict[k] = []
        map_names_new_dict[k] = []
    brain_models = None
    for subject_id in subject_ids:
        src_file = src_files.format(subject_id)
        reader = CiftiReader(src_file)
        maps = reader.get_data()
        map_names = reader.map_names()
        if brain_models is None:
            brain_models = reader.brain_models()

        # make sure that we get right copes
        for k, v in interested_copes.items():
            if v not in map_names[k]:
                raise RuntimeError("subject-{0}'s cope{1} is not {2}".format(subject_id, k, v))

        for k in interested_copes.keys():
            maps_new_dict[k].append(maps[k].copy())
            map_names_new_dict[k].append(map_names[k])
            print('Finished: merge {0}_cope{1}'.format(subject_id, k))

    for k in interested_copes.keys():
        save2cifti(trg_file.format(k, interested_copes[k]),
                   np.array(maps_new_dict[k]), brain_models, map_names_new_dict[k])
