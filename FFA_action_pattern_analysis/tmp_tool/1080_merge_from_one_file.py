if __name__ == '__main__':
    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2cifti

    # predefine some variates
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'

    reader = CiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
                         'HCP_S1200_GroupAvg_v1/S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii')
    data = reader.get_data()
    names = reader.map_names()

    subj_id_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    with open(subj_id_file) as rf:
        subj_ids = rf.read().splitlines()
    names_1080 = [_ + '_MyelinMap' for _ in subj_ids]

    indices = []
    for name in names_1080:
        if name in names:
            indices.append(names.index(name))

    data_1080 = data[indices]

    # output
    save2cifti(pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_MyelinMap_BC_MSMAll_32k_fs_LR.dscalar.nii'),
               data_1080, reader.brain_models(), names_1080)
