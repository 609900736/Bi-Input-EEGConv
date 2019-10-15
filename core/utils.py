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
    Loads the data from MAT file. MAT file would be two kinds.
    'data.mat' which contains the feature matrix in the shape
    of [nTrials, nChannels, nSamples] and 'label.mat' which
    contains the output labels as a vector. Label numbers are
    assumed to start from 0.

    Input:

        datafile        : str, load data or label from .mat file
        label           : bool, if True: load label, else: load data

    Output:

        data or label   : ndarray
    """
    print("Loading data from %s" % (datafile))
    dataMat = sio.loadmat(datafile, mat_dtype=True)
    if label:
        print("Data loading complete. Shape is %r" %
              (dataMat['classlabel'].shape, ))
        return dataMat['classlabel'] - 1
    else:  # [nChannels, nSamples, nTrials]
        dataMat['s'] = dataMat['s'].swapaxes(1, 2)
        dataMat['s'] = dataMat['s'].swapaxes(0, 1)
        print("Data loading complete. Shape is %r" % (dataMat['s'].shape, ))
        return dataMat['s']  # [nTrials, nChannels, nSamples]


def gen_images(locs,
               features,
               nGridpoints,
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
        features    : ndarray, Feature matrix as [nChannels, nSamples, nColors] Features are as columns.
                      Features corresponding to each frequency band are concatenated.
        nGridpoints : int, Number of pixels in the output images
        normalize   : bool, Flag for whether to normalize each band over all samples
        augment     : bool, Flag for generating augmented images
        pca         : bool, Flag for PCA based data augmentation
        stdmult     : float, Multiplier for std of added noise
        nComponents : int, Number of components in PCA to retain for augmentation
        edgeless    : bool, If True generates edgeless images by adding artificial channels
                      at four corners of the image with value = 0 (default = False).
        
    Output:

        interp      : ndarray, Tensor of size [nSamples, H, W, nColors] containing generated images.
    """
    feat_array_temp = []
    nChannels = features.shape[0]  # Number of electrodes
    nColors = features.shape[2]
    for c in range(nColors):
        feat_array_temp.append(features[:, c * nChannels:nChannels *
                                        (c + 1)])
    if augment:
        if pca:
            for c in range(nColors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c],
                                                 stdmult,
                                                 pca=True,
                                                 nComponents=nComponents)
        else:
            for c in range(nColors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c],
                                                 stdmult,
                                                 pca=False,
                                                 nComponents=nComponents)
    nSamples = features.shape[0]

    # Interpolate the values
    grid_x, grid_y = np.mgrid[min(locs[:, 0]):max(locs[:, 0]):nGridpoints * 1j,
                              min(locs[:, 1]):max(locs[:, 1]):nGridpoints * 1j]
    interp = []
    for c in range(nColors):
        interp.append(np.zeros([nSamples, nGridpoints, nGridpoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs,
                         np.array([[min_x, min_y], [min_x, max_y],
                                   [max_x, min_y], [max_x, max_y]]),
                         axis=0)
        for c in range(nColors):
            feat_array_temp[c] = np.append(feat_array_temp[c],
                                           np.zeros((nSamples, 4)),
                                           axis=1)

    # Interpolating
    for i in range(nSamples):
        for c in range(nColors):
            interp[c][i, :, :] = griddata(locs,
                                          feat_array_temp[c][i, :],
                                          (grid_x, grid_y),
                                          method='cubic',
                                          fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')

    # Normalizing
    for c in range(nColors):
        if normalize:
            interp[c][~np.isnan(interp[c])] = \
                scale(interp[c][~np.isnan(interp[c])])
        interp[c] = np.nan_to_num(interp[c])

    interp = np.swapaxes(np.asarray(interp), 0,
                         1)  # swap axes to have [nSamples, H, W, nColors]
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


def load_or_generate_images(filepath=None, locspath=None, average_image=3):
    """
    Generates EEG images

    Input:

        filepath: str, path of images data file, default is None
        locspath: str, path of locations data file, default is None

    Output:

        data    : ndarray, Tensor of size [nTrials, nSamples, H, W, nColors] containing generated images.
    """
    if filepath is None:
        filepath = ''
    if locspath is None:
        locspath = 'data/Neuroscan_locs_orig.mat'
    print('-' * 100)
    print('Loading original data...')
    locs = sio.loadmat(locspath)
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    data = load_data(filepath)

    # Class labels should start from 0
    f, t, Zxx = signal.stft(data, fs=250, window='hann')
    feats = np.abs(Zxx)
    labels = load_data(filepath)

    if average_image == 1:  # for CNN only
        if os.path.exists(filepath + 'images_average.mat'):
            images_average = sio.loadmat(
                filepath + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp = feats[:, i * 192:(i + 1) *
                                 192]  # each window contains 64*3=192 data
                else:
                    temp += feats[:, i * 192:(i + 1) * 192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d),
                                        av_feats,
                                        32,
                                        normalize=False)
            sio.savemat(filepath + 'images_average.mat',
                        {'images_average': images_average})
            print('Saving images_average done!')
        del feats
        images_average = images_average[np.newaxis, :]
        print('The shape of images_average.shape', images_average.shape)
        return images_average, labels
    elif average_image == 2:  # for mulit-frame model such as LSTM
        if os.path.exists(filepath + 'images_timewin.mat'):
            images_timewin = sio.loadmat(
                filepath + 'images_timewin.mat')['images_timewin']
            print('\n')
            print('Load images_timewin done!')
        else:
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(np.array(locs_2d),
                           feats[:, i * 192:(i + 1) * 192],
                           32,
                           normalize=False)
                for i in range(feats.shape[1] // 192)
            ])
            sio.savemat(filepath + 'images_timewin.mat',
                        {'images_timewin': images_timewin})
            print('Saving images for all time windows done!')
        del feats
        print('The shape of images_timewin is',
              images_timewin.shape)  # (7, 2670, 32, 32, 3)
        return images_timewin, labels
    else:
        if os.path.exists(filepath + 'images_average.mat'):
            images_average = sio.loadmat(
                filepath + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp = feats[:, i * 192:(i + 1) * 192]
                else:
                    temp += feats[:, i * 192:(i + 1) * 192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d),
                                        av_feats,
                                        32,
                                        normalize=False)
            sio.savemat(filepath + 'images_average.mat',
                        {'images_average': images_average})
            print('Saving images_average done!')

        if os.path.exists(filepath + 'images_timewin.mat'):
            images_timewin = sio.loadmat(
                filepath + 'images_timewin.mat')['images_timewin']
            print('\n')
            print('Load images_timewin done!')
        else:
            print('\n')
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(np.array(locs_2d),
                           feats[:, i * 192:(i + 1) * 192],
                           32,
                           normalize=False)
                for i in range(feats.shape[1] // 192)
            ])
            sio.savemat(filepath + 'images_timewin.mat',
                        {'images_timewin': images_timewin})
            print('Saving images for all time windows done!')

        del feats
        images_average = images_average[np.newaxis, :]
        print('The shape of labels.shape', labels.shape)
        print('The shape of images_average.shape',
              images_average.shape)  # (1, 2670, 32, 32, 3)
        print('The shape of images_timewin is',
              images_timewin.shape)  # (7, 2670, 32, 32, 3)
        return images_average, images_timewin, labels


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