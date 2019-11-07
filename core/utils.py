# -*- coding:utf-8 -*-

import os
import pywt
import math as m
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# from functools import reduce


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical

    Input:

        x: float, X coordinate
        y: float, Y coordinate
        z: float, Z coordinate

    Output:

        radius, elevation, azimuth: float -> tuple, Transformed polar coordinates
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian

    Input:
   
        theta   : float, angle value
        rho     : float, radius value
    
    Output:

        X, Y    : float -> tuple, projected coordinates
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in
    3D Cartesian Coordinates. Imagine a plane being placed against
    (tangent to) a globe. If a light source inside the globe projects
    the graticule onto the plane the result would be a planar, or
    azimuthal, map projection.

    Input:
        
        pos     : list or tuple, position in 3D Cartesian coordinates [x, y, z]
        
    Output:

        X, Y    : float -> tuple, projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def load_data(datafile, label=True):
    """
    Loads the data from MAT file. 
    
    MAT file would be two kinds. `'*.mat'` which contains the feature 
    matrix in the shape of `[nTrials, nChannels, nSamples]` and 
    `'*_label.mat'` which contains the output labels as a vector. 
    Label numbers are assumed to start from 0.

    Parameters
    ----------
    ```txt
    datafile        : str, load data or label from *.mat file (* in '*.mat' 
                      and '*_label.mat' are the same, pls let datafile = '*.mat')
    label           : bool, if True: load label, else: load data
    ```
    Returns
    -------
    ```txt
    data or label   : ndarray
    ```
    """
    print("Loading data from %s" % (datafile))
    if label:
        dataMat = sio.loadmat(datafile[:-4] + '_label.mat', mat_dtype=True)
        print("Data loading complete. Shape is %r" %
              (dataMat['classlabel'].shape, ))
        # Class labels should start from 0
        return dataMat['classlabel'] - 1
    else:  # [nChannels, nSamples, nTrials]
        dataMat = sio.loadmat(datafile, mat_dtype=True)
        dataMat['s'] = dataMat['s'].swapaxes(1, 2)
        dataMat['s'] = dataMat['s'].swapaxes(0, 1)
        print("Data loading complete. Shape is %r" % (dataMat['s'].shape, ))
        return dataMat['s']  # [nTrials, nChannels, nSamples]


def gen_images(locs,
               features,
               H,
               W,
               mode='interpolation',
               normalize=True,
               augment=False,
               pca=False,
               stdmult=0.1,
               nComponents=2,
               edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    Input:

        locs        : ndarray, An array with shape [nChannels, 2] containing X, Y coordinates
                      for each electrode.
        features    : ndarray, Feature matrix as [nSamples, nChannels, nColors] Features are as columns.
                      Features corresponding to each frequency band are concatenated.
        H           : int, Number of pixels in the output images height
        W           : int, Number of pixels in the output images width
        mode        : str, Mode of generation, can be choose between 'interpolation' and 'raw', 
                      if mode == 'raw', locs, H, W and edgeless will be invalid, default is 'interpolation'
        normalize   : bool, Flag for whether to normalize each band over all samples
        augment     : bool, Flag for generating augmented images
        pca         : bool, Flag for PCA based data augmentation
        stdmult     : float, Multiplier for std of added noise
        nComponents : int, Number of components in PCA to retain for augmentation
        edgeless    : bool, If True generates edgeless images by adding artificial channels
                      at four corners of the image with value = 0, default is False.
        
    Output:

        interp      : ndarray, Tensor with size of [nSamples, H, W, nColors] containing generated images.
    """
    feat_array_temp = features
    nChannels = feat_array_temp.shape[1]  # Number of electrodes
    nColors = feat_array_temp.shape[2]

    if augment:
        if pca:
            for c in range(nColors):
                feat_array_temp[:, :, c] = augment_EEG(
                    feat_array_temp[:, :, c],
                    stdmult,
                    pca=True,
                    nComponents=nComponents)
        else:
            for c in range(nColors):
                feat_array_temp[:, :, c] = augment_EEG(
                    feat_array_temp[:, :, c],
                    stdmult,
                    pca=False,
                    nComponents=nComponents)
    nSamples = feat_array_temp.shape[0]

    # Interpolate the values
    if mode == 'interpolation':
        grid_x, grid_y = np.mgrid[min(locs[:, 0]):max(locs[:, 0]):W * 1j,
                                  min(locs[:, 1]):max(locs[:, 1]):H * 1j]
    interp = []
    for c in range(nColors):
        if mode == 'interpolation':
            interp.append(np.zeros([nSamples, H, W]))
        elif mode == 'raw':
            interp.append(np.zeros([nSamples, 6, 7]))
        else:
            raise ValueError(
                'gen_images: mode can only be one of \'interpolation\' and \'raw\''
            )

    # Generate edgeless images
    if edgeless and mode == 'interpolation':
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs,
                         np.array([[min_x, min_y], [min_x, max_y],
                                   [max_x, min_y], [max_x, max_y]]),
                         axis=0)
        for c in range(nColors):
            feat_array_temp[:, :, c] = np.append(feat_array_temp[:, :, c],
                                                 np.zeros((nSamples, 4)),
                                                 axis=1)

    # Generating
    for i in range(nSamples):
        for c in range(nColors):
            if mode == 'interpolation':
                interp[c][i, :, :] = griddata(locs,
                                              feat_array_temp[i, :, c],
                                              (grid_x, grid_y),
                                              method='cubic',
                                              fill_value=np.nan).T
            elif mode == 'raw':
                interp[c][i, 0, 3:4] = feat_array_temp[i, 0:1, c]
                interp[c][i, 1, 1:6] = feat_array_temp[i, 1:6, c]
                interp[c][i, 2, 0:7] = feat_array_temp[i, 6:13, c]
                interp[c][i, 3, 1:6] = feat_array_temp[i, 13:18, c]
                interp[c][i, 4, 2:5] = feat_array_temp[i, 18:21, c]
                interp[c][i, 5, 3:4] = feat_array_temp[i, 21:22, c]
            else:
                raise ValueError(
                    'gen_images: mode can only be one of \'interpolation\' and \'raw\''
                )
        print('Generating {0:0>4d}/{1:0>4d}\r'.format(i + 1, nSamples),
              end='\r')
    print()

    # Normalizing
    for c in range(nColors):
        if normalize:
            interp[c][~np.isnan(interp[c])] = \
                scale(interp[c][~np.isnan(interp[c])])
        interp[c] = np.nan_to_num(interp[c])

    interp = np.asarray(interp)
    # swap axes to have [nSamples, H, W, nColors]
    interp = np.swapaxes(interp, 0, 1)
    interp = np.swapaxes(interp, 1, 2)
    interp = np.swapaxes(interp, 2, 3)

    return interp


def augment_EEG(data, stdmult, pca=False, nComponents=2):
    """
    Augment data by adding normal noise to each feature.

    Input:

        data        : EEG feature data as a matrix (nSamples, nFeatures)
        stdmult     : Multiplier for std of added noise
        pca         : if True will perform PCA on data and add noise proportional to PCA components.
        nComponents : Number of components to consider when using PCA.

    Output:

        augData     : Augmented data as a matrix (nSamples, nFSeatures)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(nComponents=nComponents)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdmult,
                                  size=pca.nComponents) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape(
                (nComponents, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(
                scale=stdmult * np.std(feat), size=feat.size)
    return augData


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
        return [(data[trainIndices],
                 np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices],
                 np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices],
                 np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices],
                 np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices],
                 np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices],
                 np.squeeze(labels[testIndices]).astype(np.int32))]


def load_or_generate_images(filepath,
                            locspath=None,
                            beg=0,
                            end=4,
                            srate=250,
                            mode='raw',
                            averageImages=1,
                            H=30,
                            W=35):
    """
    Generates EEG images

    Input:

        filepath        : str, path of images data file
        locspath        : str, path of locations data file, default is None
        beg             : num, second when imegery tasks begins, default is 0
        end             : num, second when imegery tasks ends, default is 4
        srate           : int, the sample rate of raw data, default is 250 
        mode            : str, should be one of strings among 'raw', 'topography', 'energy', and 'envelope', default is 'raw'
        averageImages   : int, length of window to mix images in time dimension, like AveragePooling2D(1, averageImages), default is 1
        H               : int, 
        W               : int, 

    Output:

        imagesData      : ndarray, Tensor of size [nTrials, nSamples, H, W, nColors] containing generated images

    *********************************************************
    
        type num means int or float
    """
    if locspath is None:
        locspath = 'data/22scan_locs.mat'
    print('-' * 100)
    print('Loading data...')
    locs_3d = load_locs(locspath)
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    if mode == 'raw':
        if os.path.exists(filepath[:-4] + '_raw_' + str(averageImages) +
                          '.mat'):
            images_average = sio.loadmat(filepath[:-4] + '_raw_' +
                                         str(averageImages) +
                                         '.mat')['images_average']
            print('Load images_average done!')
        else:
            print('Generating average images over time windows...')
            feats = load_data(filepath, label=False)
            # feats = bandpassfilter(feats)
            feats = feats[:, :, :, np.newaxis]
            images_average = []
            for n in range(feats.shape[0]):
                print('Generate trial {:0>3d}'.format(n + 1))
                av_feats = []
                for i in range(feats.shape[2] // averageImages):
                    av_feats.append(
                        np.average(feats[n, :, i * averageImages:(i + 1) *
                                         averageImages, :],
                                   axis=1))
                images_average.append(
                    gen_images(None,
                               np.asarray(av_feats),
                               None,
                               None,
                               mode='raw',
                               normalize=False))
            images_average = np.asarray(images_average)
            sio.savemat(filepath[:-4] + '_raw_' + str(averageImages) + '.mat',
                        {'images_average': images_average})
            print()
            print('Saving images_average done!')
            del feats
        images_average = images_average[:,
                                        m.floor(beg *
                                                srate):m.ceil(end *
                                                              srate), :, :, :]
        print('The shape of images_average.shape', images_average.shape)
        pass
    elif mode == 'topography':
        if os.path.exists(filepath[:-4] + '_topography_' + str(H) + '_' +
                          str(W) + '_' + str(averageImages) + '.mat'):
            images_average = sio.loadmat(filepath[:-4] + '_topography_' +
                                         str(H) + '_' + str(W) + '_' +
                                         str(averageImages) +
                                         '.mat')['images_average']
            print('Load images_average done!')
        else:
            print('Generating average images over time windows...')
            # feats = load_or_gen_interestingband_data(filepath,
            #                                          beg=beg,
            #                                          end=end,
            #                                          srate=srate)
            feats = load_data(filepath, label=False)
            # feats = bandpassfilter(feats)
            feats = feats[:, :, :, np.newaxis]
            images_average = []
            for n in range(feats.shape[0]):
                print('Generate trial {:0>3d}'.format(n + 1))
                av_feats = []
                for i in range(feats.shape[2] // averageImages):
                    av_feats.append(
                        np.average(feats[n, :, i * averageImages:(i + 1) *
                                         averageImages, :],
                                   axis=1))
                images_average.append(
                    gen_images(np.asarray(locs_2d),
                               np.asarray(av_feats),
                               H,
                               W,
                               normalize=False))
            images_average = np.asarray(images_average)
            sio.savemat(
                filepath[:-4] + '_topography_' + str(H) + '_' + str(W) + '_' +
                str(averageImages) + '.mat',
                {'images_average': images_average})
            print()
            print('Saving images_average done!')
            del feats
        images_average = images_average[:,
                                        m.floor(beg *
                                                srate):m.ceil(end *
                                                              srate), :, :, :]
        print('The shape of images_average.shape', images_average.shape)
        pass
    elif mode == 'energy':
        if os.path.exists(filepath[:-4] + '_energy_' + str(H) + '_' + str(W) +
                          '_' + str(averageImages) + '.mat'):
            images_average = sio.loadmat(filepath[:-4] + '_energy_' + str(H) +
                                         '_' + str(W) + '_' +
                                         str(averageImages) +
                                         '.mat')['images_average']
            print('Load images_average done!')
        else:
            print('Generating average images over time windows...')
            f, t, Zxx = signal.stft(load_or_gen_interestingband_data(filepath),
                                    fs=250,
                                    window='hann',
                                    axis=2)
            feats = np.abs(Zxx)
            images_average = []
            for n in range(feats.shape[0]):
                print('Generate trial {:0>3d}'.format(n + 1))
                av_feats = []
                for i in range(feats.shape[2] // averageImages):
                    av_feats.append(
                        np.average(feats[n, :, i * averageImages:(i + 1) *
                                         averageImages, :],
                                   axis=1))
                images_average.append(
                    gen_images(np.asarray(locs_2d),
                               np.asarray(av_feats),
                               32,
                               normalize=False))
            images_average = np.asarray(images_average)
            sio.savemat(
                filepath[:-4] + '_energy_' + str(H) + '_' + str(W) + '_' +
                str(averageImages) + '.mat',
                {'images_average': images_average})
            print('Saving images_average done!')
            del feats
        images_average = images_average[:,
                                        m.floor(beg *
                                                srate):m.ceil(end *
                                                              srate), :, :, :]
        print('The shape of images_average.shape', images_average.shape)
        pass
    elif mode == 'envelope':
        # signal.hilbert(load_data(filepath), )
        pass
    else:
        raise ValueError(
            'load_or_generate_images: mode should be one of strings among \'raw\', \'topography\', \'energy\', and \'envelope\''
        )
        pass
    labels = load_data(filepath[:-4] + '_label.mat')
    return images_average, labels


def filterbank(data, srate=250, start=4, stop=38, window=4, step=2):
    '''
    Process raw data with filter-bank.

    Input:

        data    : ndarray, raw data, shapes as [nTrials, nChannels, nSamples]
        srate   : int, the sample rate of raw data, default is 250
        start   : int, frequency where the filter-bank begins, default is 4
        stop    : int, frequency where the filter-bank ends, default is 38
        window  : int, the bandwidth of one filter in the filter-bank, default is 4
        step    : int, the interval of each neighbouring filter in the filter-bank, default is 2

    Output:

        FBdata  : ndarray, data after filter-bank, shapes as [nTrials, nChannels, nSamples, nColors]
    '''
    nTrials, nChannels, nSamples = data.shape
    FBdata = []
    for beg in range(start, stop - window + 1, step):
        end = beg + window
        b, a = signal.butter(4, [beg / srate * 2, end / srate * 2], 'bandpass')
        FBdata.append(signal.filtfilt(b, a, data, axis=-1))
    #now np.array(FBdata) shapes as[nColors, nTrials, nChannels, nSamples]
    FBdata = np.swapaxes(np.array(FBdata), 0, 1)
    FBdata = np.swapaxes(FBdata, 1, 2)
    FBdata = np.swapaxes(FBdata, 2, 3)
    print("Data filterbank complete. Shape is %r." % (FBdata.shape, ))
    return FBdata


def load_or_gen_filterbank_data(filepath,
                                beg=0,
                                end=4,
                                srate=250,
                                start=4,
                                stop=38,
                                window=4,
                                step=4):
    '''
    load or generate data with filter-bank.

    Input:

        filepath: str, path of raw data file, and data shape is [nTrials, nChannels, nSamples]
        beg     : num, second when imegery tasks begins
        end     : num, second when imegery tasks ends
        srate   : int, the sample rate of raw data, default is 250
        start   : int, frequency where the filter-bank begins, default is 4
        stop    : int, frequency where the filter-bank ends, default is 38
        window  : int, the bandwidth of one filter in the filter-bank, default is 4
        step    : int, the interval of each neighbouring filter in the filter-bank, default is 2

    Output:
    
        FBdata  : ndarray, data after filter-bank, shapes as [nTrials, nChannels, nSamples, nColors]

    *********************************************************
    
        type num means int or float
    '''
    if os.path.exists(filepath[:-4] + '_fb.mat'):
        print('Loading data from %s' % (filepath[:-4] + '_fb.mat'))
        data = sio.loadmat(filepath[:-4] + '_fb.mat')['fb']
        print('Load filterbank data complete. Shape is %r.' % (data.shape, ))
    else:
        data = filterbank(load_data(filepath, label=False),
                          srate=srate,
                          start=start,
                          stop=stop,
                          window=window,
                          step=step)
        data = data[:, :, beg * srate:end * srate, :]
        print('Load filterbank data complete. Shape is %r.' % (data.shape, ))
        sio.savemat(filepath[:-4] + '_fb.mat', {'fb': data})
        print('Save filterbank data[\'fb\'] complete. To %s' %
              (filepath[:-4] + '_fb.mat'))

    return data


def load_locs(filepath=None):
    '''
    load data of electrodesr' 3D location.

    Input:

        filepath: str, path of electrodes' 3D location data file, default is None

    Output:

        locs    : ndarray, data of electrodes' 3D location, shapes as [nChannels, 3]
    '''
    if filepath is None:
        filepath = os.path.join('data', '22scan_locs.mat')
    locs = sio.loadmat(filepath)['A']
    return locs


def interestingband(data, srate=250):
    '''
    Filter raw signal to five interesting bands - theta, alpha, beta, low gamma, high gamma.
    theta: 4-8Hz
    alpha: 8-13Hz
    beta: 14-30Hz
    low gamma: 30-50Hz
    high gamma: 71-91Hz

    Input:

        data    : ndarray, raw data, shapes as [nTrials, nChannels, nSamples]
        srate   : int, the sample rate of raw data, default is 250

    Output:

        IBdata  : ndarray, data after filter-bank, shapes as [nTrials, nChannels, nSamples, nColors]
    '''
    eps = 1e-9
    IBdata = []
    b, a = signal.butter(1, [4, 8], 'bandpass', fs=srate)  # theta
    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    IBdata.append(
        signal.filtfilt(b,
                        a,
                        data,
                        axis=-1,
                        method='gust',
                        irlen=approx_impulse_len))
    b, a = signal.butter(2, [8, 13], 'bandpass', fs=srate)  # alpha
    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    IBdata.append(
        signal.filtfilt(b,
                        a,
                        data,
                        axis=-1,
                        method='gust',
                        irlen=approx_impulse_len))
    b, a = signal.butter(3, [14, 30], 'bandpass', fs=srate)  # beta
    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    IBdata.append(
        signal.filtfilt(b,
                        a,
                        data,
                        axis=-1,
                        method='gust',
                        irlen=approx_impulse_len))
    b, a = signal.butter(4, [30, 50], 'bandpass', fs=srate)  # low gamma
    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    IBdata.append(
        signal.filtfilt(b,
                        a,
                        data,
                        axis=-1,
                        method='gust',
                        irlen=approx_impulse_len))
    b, a = signal.butter(4, [71, 91], 'bandpass', fs=srate)  # high gamma
    z, p, k = signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    IBdata.append(
        signal.filtfilt(b,
                        a,
                        data,
                        axis=-1,
                        method='gust',
                        irlen=approx_impulse_len))
    # now np.array(IBdata) shapes as[nColors, nTrials, nChannels, nSamples]
    IBdata = np.swapaxes(np.array(IBdata), 0, 1)
    IBdata = np.swapaxes(IBdata, 1, 2)
    IBdata = np.swapaxes(IBdata, 2, 3)
    print("Data filterbank complete. Shape is %r." % (IBdata.shape, ))
    return IBdata


def load_or_gen_interestingband_data(filepath, beg=0, end=4, srate=250):
    '''
    load or generate data with interesting-band filters.

    Input:

        filepath: str, path of raw data file, and data shape is [nTrials, nChannels, nSamples]
        beg     : num, second when imegery tasks begins
        end     : num, second when imegery tasks ends
        srate   : int, the sample rate of raw data, default is 250

    Output:

        IBdata  : ndarray, data after interesting-band filters, shapes as [nTrials, nChannels, nSamples, nColors]

    *********************************************************

        type num means int or float
    '''
    if os.path.exists(filepath[:-4] + '_ib.mat'):
        print('Loading data from %s' % (filepath[:-4] + '_ib.mat'))
        data = sio.loadmat(filepath[:-4] + '_ib.mat')['ib']
        print('Load interestingband data complete. Shape is %r.' %
              (data.shape, ))
    else:
        data = interestingband(load_data(filepath, label=False), srate=srate)
        data = data[:, :, beg * srate:end * srate, :]
        print('Load interestingband data complete. Shape is %r.' %
              (data.shape, ))
        sio.savemat(filepath[:-4] + '_ib.mat', {'ib': data})
        print('Save interestingband data[\'ib\'] complete. To %s' %
              (filepath[:-4] + '_ib.mat'))

    return data


def highpassfilter(data, Wn=4, srate=250):
    b, a = signal.butter(4, Wn=Wn, btype='highpass', fs=srate)
    new_data = []
    for e in data:
        new_data.append(signal.filtfilt(b, a, e))
    return np.asarray(new_data)


def bandpassfilter(data, Wn=[.5, 100], srate=250):
    b, a = signal.butter(4, Wn=Wn, btype='bandpass', fs=srate)
    new_data = []
    for e in data:
        new_data.append(signal.filtfilt(b, a, e))
    return np.asarray(new_data)


# In order to gain energy-spectrum, cwt, stft, hht, and envelope is considered
def cwt(data):
    signal.cwt()  # lack of defined wavelets, return conv
    pywt.cwt()  # return coef
    pywt.dwt()
    pywt.wavedec()
    signal.stft()
    signal.hilbert()
    return data


if __name__ == '__main__':
    print(pywt.wavelist())
    print(pywt.ContinuousWavelet('gaus8'))
    x = np.linspace(0, 4, 1000)
    y = (1 + 0.5 * np.cos(2 * np.pi * 5 * x)
         ) * np.cos(2 * np.pi * 50 * x + 0.5 * np.sin(2 * np.pi * 10 * x))
    plt.plot(x, y)  # doctest: +SKIP
    coef, freqs = pywt.cwt(y,
                           np.arange(1, 129),
                           'gaus4',
                           sampling_period=1.0 / 250)
    plt.matshow(abs(coef))  # doctest: +SKIP
    print(freqs)
    plt.figure()
    plt.contourf(x, freqs, abs(coef))
    f, t, Zxx = signal.stft(y, fs=250, window='hann')
    plt.figure()
    print(f)
    plt.pcolormesh(t,
                   f,
                   np.abs(Zxx),
                   vmin=np.min(np.abs(Zxx)),
                   vmax=np.max(np.abs(Zxx)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # t = np.linspace(-1, 1, 200, endpoint=False)
    # sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    # widths = np.arange(1, 31)
    # cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus8')
    # plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
    # plt.show() # doctest: +SKIP
    #t = np.linspace(0, 1, 2000, False)  # 1 second
    #sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)    # 构造10hz和20hz的两个信号
    #sig = np.array([[sig,sig],[sig,sig]])
    #print(sig.shape)
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #ax1.plot(t, sig[0,0,:])
    #ax1.set_title('10 Hz and 20 Hz sinusoids')
    #ax1.axis([0, 1, -2, 2])

    #b, a = signal.butter(4, [14,30], 'bandpass', fs=2000, output='ba')     #采样率为1000hz，带宽为15hz，输出ba
    #z, p, k = signal.tf2zpk(b, a)
    #eps = 1e-9
    #r = np.max(np.abs(p))
    #approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    #print(approx_impulse_len)
    #filtered = signal.filtfilt(b, a, sig, method='gust', irlen=approx_impulse_len)             #将信号和通过滤波器作用，得到滤波以后的结果。在这里sos有点像冲击响应，这个函数有点像卷积的作用。
    #ax2.plot(t, filtered[0,0,:])
    #ax2.set_title('After 15 Hz high-pass filter')
    #ax2.axis([0, 1, -2, 2])
    #ax2.set_xlabel('Time [seconds]')
    #plt.tight_layout()
    #plt.show()
    #print('BiInputConv.core.utils')