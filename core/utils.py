#coding:utf-8

import os
import math as m
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from functools import reduce


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates    [x, y, z]
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def load_data(data_file, label=False):
    """                                               
    Loads the data from MAT file. MAT file would be two kinds.
    'data.mat' which contains the feature matrix in the shape
    of [n_trials, n_channels, n_samples] and 'label.mat' which
    contains the output labels as a vector. Label numbers are
    assumed to start from 0.

    :param data_file: str  # load data or label from .mat
    :return data or label: array_like
    """
    print("Loading data from %s" % (data_file))
    dataMat = sio.loadmat(data_file, mat_dtype=True)
    if label:
        print("Data loading complete. Shape is %r" % (dataMat['classlabel'].shape,))
        return dataMat['classlabel']
    else:#[n_channels,n_samples,n_trials]
        dataMat['s'] = dataMat['s'].swapaxes(1,2)
        dataMat['s'] = dataMat['s'].swapaxes(0,1)
        print("Data loading complete. Shape is %r" % (dataMat['s'].shape,))
        return dataMat['s']


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    n_samples = features.shape[0]

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating
    for i in xrange(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    temp_interp = np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]
    return 


def augment_EEG(data, stdMult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.

    :param data: EEG feature data as a matrix (n_samples x n_features)
    :param stdMult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
    return augData


def augment_EEG_image(image, std_mult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.

    :param image: EEG feature data as a a colored image [n_samples, n_colors, W, H]
    :param std_mult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))
    for c in xrange(image.shape[1]):
        reshData = np.reshape(data['featMat'][:, c, :, :], (data['featMat'].shape[0], -1))
        if pca:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=True, n_components=n_components)
        else:
            augData[:, c, :] = augment_EEG(reshData, std_mult, pca=False)
    return np.reshape(augData, data['featMat'].shape)


def reformatInput(data, labels, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]
    # Shuffling training data
    # shuffledIndices = np.random.permutation(len(trainIndices))
    # trainIndices = trainIndices[shuffledIndices]
    if data.ndim == 4:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]


def load_or_generate_images(file_path, average_image=3):
    """
    Generates EEG images
    :param average_image: average_image 1 for CNN model only, 2 for multi-frame model 
                        sucn as lstm, 3 for both.

    :return:            Tensor of size [window_size, samples, W, H, channel] containing generated
                        images.
    """
    print('-'*100)
    print('Loading original data...')
    locs = scipy.io.loadmat('data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    # Class labels should start from 0
    data = load_data('SampleData/FeatureMat_timeWin.mat')   # 2670*1344 和 2670*1
    labels = load_data('SampleData/FeatureMat_timeWin.mat')
    

    if average_image == 1:   # for CNN only
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp  = feats[:, i*192:(i+1)*192]    # each window contains 64*3=192 data
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')
        
        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of images_average.shape', images_average.shape)
        return images_average, labels
    
    elif average_image == 2:    # for mulit-frame model such as LSTM
        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')
        
        del feats
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_timewin, labels
    
    else:
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp = feats[:, i*192:(i+1)*192]
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')

        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('\n')
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')

        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of labels.shape', labels.shape)
        print('The shape of images_average.shape', images_average.shape)    # (1, 2670, 32, 32, 3)
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_average, images_timewin, labels


def filterbank(data, srate=250, start=4, stop=38, window=4, step=2):
    '''
    Process raw data with filter-bank.

    Input:
        data    : np.array, raw data, shapes as [n_trials, n_channels, n_samples]
        srate   : int, the sample rate of raw data, default is 250
        start   : int, frequency where the filter-bank begins, default is 4
        stop    : int, frequency where the filter-bank ends, default is 38
        window  : int, the bandwidth of one filter in the filter-bank, default is 4
        step    : int, the interval of each neighbouring filter in the filter-bank, default is 2

    Output:
        FBdata  : np.array, data after filter-bank, shapes as [n_trials, n_colors, n_channels, n_samples]
    '''
    n_trials, n_channels, n_samples = data.shape
    FBdata = []
    for beg in range(start, stop - window + 1, step):
        end = beg + window
        b, a = signal.butter(4, [beg, end]/srate*2, 'bandpass')
        FBdata.append(signal.filtfilt(b, a, data, axis=1, method='gust', irlen=50))
    #now np.array(FBdata) shapes as[n_colors, n_trials, n_channels, n_samples]
    FBdata = np.swapaxes(np.array(FBdata), 0, 1)
    return FBdata


if __name__=='__main__':
    print('BiInputConv.core.utils')