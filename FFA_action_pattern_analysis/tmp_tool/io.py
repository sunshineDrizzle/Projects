import numpy as np
import nibabel as nib


def read_nifti(fpath):
    """
    read Nifti file data
    If the data in the file is a 1D array, it will be regard as a surface map.
    If the data in the file is a 2D array, each row of it will be regard as a surface map.
    If the data in the file is a 3D array, it will be raveled as a surface map.
    If the data in the file is a 4D array, the first 3 dimensions will be raveled as a surface map,
    and the forth dimension is the number of surface maps.
    If the number of dimension is larger than 4, an error will be thrown.
    :param fpath: str
    :return: data: numpy array
    """
    img = nib.load(fpath)

    # get data
    data = img.get_data()
    if data.ndim <= 2:
        data = np.atleast_2d(data)
    elif data.ndim == 3:
        data = np.atleast_2d(np.ravel(data, order='F'))
    elif data.ndim == 4:
        _data = []
        for idx in range(data.shape[3]):
            _data.append(np.ravel(data[..., idx], order='F'))
        data = np.array(_data)
    else:
        raise ValueError("The number of dimension of data array is larger than 4.")

    return data
