import numpy as np

from commontool.io.io import CiftiReader


def extract_run_series(src_file, hemis, brain_structures, trg_regions_lr):
    region_series_list = []
    reader = CiftiReader(src_file)
    for hemi in hemis:
        series = reader.get_data(brain_structures[hemi], True)
        for trg_region in trg_regions_lr[hemi]:
            region_series_list.append(np.nanmean(series[:, trg_region], 1))

    run_series = np.array(region_series_list)
    return run_series


if __name__ == '__main__':
    import os
    import subprocess
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CsvReader

    # set paths
    project_dir = '/home/ubuntu/s3/hcp'
    series_files = pjoin(project_dir, '{subject}/MNINonLinear/Results/rfMRI_REST{sess}_{phase}/rfMRI_REST{sess}_{phase}_Atlas_MSMAll_hp2000_clean.dtseries.nii')
    group_labels_file = 'group_labels'
    subject_ids_file = 'subject_id'
    roi_files = '{}{}_FFA.nii.gz'
    mask_file_l = 'PAM_z165_p025_ROI_l.nii.gz'
    mask_file_r = 'PAM_z165_p025_ROI_r.nii.gz'
    mask_labelconfig_l = 'PAM_z165_p025_ROI_labelconfig_l.csv'
    mask_labelconfig_r = 'PAM_z165_p025_ROI_labelconfig_r.csv'

    # prepare some variables
    sessions = ('1', '2')
    phases = ('LR', 'RL')
    hemis = ('lh', 'rh')
    brain_structures = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }

    # read text file
    subject_ids = open(subject_ids_file).read().splitlines()
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)
    group_labels_uniq = np.unique(group_labels)

    # prepare target regions
    trg_regions_lr = dict()
    mask_l = nib.load(mask_file_l).get_data().ravel()
    mask_r = nib.load(mask_file_r).get_data().ravel()
    trg_regions_lr['lh'] = [np.where(mask_l == i)[0] for i in np.unique(mask_l) if i != 0]
    trg_regions_lr['rh'] = [np.where(mask_r == i)[0] for i in np.unique(mask_r) if i != 0]
    label_names_lr = dict()
    label_names_lr['lh'] = CsvReader(mask_labelconfig_l).to_dict()['label_name']
    label_names_lr['rh'] = CsvReader(mask_labelconfig_r).to_dict()['label_name']
    for group_label in group_labels_uniq:
        for hemi in hemis:
            roi_file = roi_files.format(hemi[0], group_label)
            roi_name = roi_file.split('.')[0]
            roi_data = nib.load(roi_file).get_data().ravel()
            trg_regions_lr[hemi].append(np.where(roi_data != 0)[0])
            label_names_lr[hemi].append(roi_name)
            roi_data_uniq = np.unique(roi_data).astype(np.uint8)
            for roi_label in roi_data_uniq:
                if roi_label != 0:
                    trg_regions_lr[hemi].append(np.where(roi_data == roi_label)[0])
                    label_names_lr[hemi].append(roi_name + str(roi_label))

    # calculation
    log_writer = open('extract_roi_tseries_log', 'w+')
    subject_num = len(subject_ids)
    for count, subject in enumerate(subject_ids, 1):
        print('{}/{}'.format(count, subject_num))
        if os.path.exists(subject):
            continue
        
        series_file_1LR = series_files.format(subject=subject, sess='1', phase='LR')
        series_file_1RL = series_files.format(subject=subject, sess='1', phase='RL')
        series_file_2LR = series_files.format(subject=subject, sess='2', phase='LR')
        series_file_2RL = series_files.format(subject=subject, sess='2', phase='RL')
        pass_subject = False
        if not os.path.exists(series_file_1LR):
            log_writer.writelines('Path-{} does not exist!\n'.format(series_file_1LR))
            pass_subject = True
        if not os.path.exists(series_file_1RL):
            log_writer.writelines('Path-{} does not exist!\n'.format(series_file_1RL))
            pass_subject = True
        if not os.path.exists(series_file_2LR):
            log_writer.writelines('Path-{} does not exist!\n'.format(series_file_2LR))
            pass_subject = True
        if not os.path.exists(series_file_2RL):
            log_writer.writelines('Path-{} does not exist!\n'.format(series_file_2RL))
            pass_subject = True
        if pass_subject:
            continue

        try:
            run_series_1LR = extract_run_series(series_file_1LR, hemis, brain_structures, trg_regions_lr)
            time_point_num = run_series_1LR.shape[1]
            if time_point_num != 1200:
                log_writer.writelines('{}: time_point_num is not 1200, but is {}'.format(series_file_1LR, time_point_num))
                continue
        except BaseException as err:
            log_writer.writelines('{} meet error: {}\n'.format(series_file_1LR, err))
            continue
        try:
            run_series_1RL = extract_run_series(series_file_1RL, hemis, brain_structures, trg_regions_lr)
            time_point_num = run_series_1RL.shape[1]
            if time_point_num != 1200:
                log_writer.writelines('{}: time_point_num is not 1200, but is {}'.format(series_file_1RL, time_point_num))
                continue
        except BaseException as err:
            log_writer.writelines('{} meet error: {}\n'.format(series_file_1RL, err))
            continue
        try:
            run_series_2LR = extract_run_series(series_file_2LR, hemis, brain_structures, trg_regions_lr)
            time_point_num = run_series_2LR.shape[1]
            if time_point_num != 1200:
                log_writer.writelines('{}: time_point_num is not 1200, but is {}'.format(series_file_2LR, time_point_num))
                continue
        except BaseException as err:
            log_writer.writelines('{} meet error: {}\n'.format(series_file_2LR, err))
            continue
        try:
            run_series_2RL = extract_run_series(series_file_2RL, hemis, brain_structures, trg_regions_lr)
            time_point_num = run_series_2RL.shape[1]
            if time_point_num != 1200:
                log_writer.writelines('{}: time_point_num is not 1200, but is {}'.format(series_file_2RL, time_point_num))
                continue
        except BaseException as err:
            log_writer.writelines('{} meet error: {}\n'.format(series_file_2RL, err))
            continue

        os.makedirs(subject)
        np.save(pjoin(subject, 'rfMRI_REST1_LR.npy'), run_series_1LR)
        np.save(pjoin(subject, 'rfMRI_REST1_RL.npy'), run_series_1RL)
        np.save(pjoin(subject, 'rfMRI_REST2_LR.npy'), run_series_2LR)
        np.save(pjoin(subject, 'rfMRI_REST2_RL.npy'), run_series_2RL)

        subprocess.Popen('sudo rm -rfv /tmp/hcp-openaccess/*', shell=True)
    log_writer.close()
    arr_shape_info = 'array_shape,(region_num time_point_num)'
    region_names = ['region_name'] + label_names_lr['lh'] + label_names_lr['rh']
    region_names_info = ','.join(region_names)
    open('npy_info', 'w+').writelines('\n'.join([arr_shape_info, region_names_info]))
