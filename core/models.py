#coding:utf-8

#import tensorflow.python.keras.api._v1.keras.layers
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, \
                                           Conv2D, \
                                           Conv3D, \
                                           Concatenate, \
                                           BatchNormalization, \
                                           AveragePooling1D, \
                                           AveragePooling2D, \
                                           AveragePooling3D, \
                                           MaxPooling2D, \
                                           MaxPooling3D, \
                                           SeparableConv2D, \
                                           DepthwiseConv2D, \
                                           Activation, \
                                           SpatialDropout2D, \
                                           SpatialDropout3D, \
                                           Dropout, \
                                           Flatten, \
                                           Lambda
from tensorflow.python.keras.constraints import max_norm, \
                                                min_max_norm, \
                                                unit_norm
from tensorflow.python.keras import backend as K
K.set_image_data_format('channels_last')


def rawEEGConvModel(Chans, Samples, Colors, dropoutRate = 0.5,
                    kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                    dtype = tf.float32, dropoutType = 'Dropout'):
    """
    for weight reusing

    see BIEEGConvNet
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'SpatialDropout3D':
        dropoutType = SpatialDropout3D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D, '
                         'SpatialDropout3D or Dropout, passed as a string.')
    # Learn from raw EEG signals
    input_s = Input(shape=(Chans, Samples, Colors), dtype=dtype)
    s = Conv2D(F1, (1, kernLength), padding = 'same', use_bias = False)(input_s)
    s = BatchNormalization(axis = -1)(s)
    s = DepthwiseConv2D((Chans, 1), use_bias = False, depth_multiplier = D,
                        depthwise_constraint = max_norm(1.))(s)
    s = BatchNormalization(axis = -1)(s)
    s = Activation('elu')(s)
    s = AveragePooling2D((1, 4))(s)
    s = dropoutType(dropoutRate)(s)
    s = SeparableConv2D(F2, (1, 16), padding = 'same', use_bias = False)(s)
    s = BatchNormalization(axis = -1)(s)
    s = Activation('elu')(s)
    s = AveragePooling2D((1, 8))(s)
    s = dropoutType(dropoutRate)(s)
    flatten = Flatten()(con)

    return Model(inputs=input_s,outputs=flatten)


def _old_rawEEGConvModel(Chans, Samples, Colors, dropoutRate = 0.5,
                         kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                         dtype = tf.float32, dropoutType = 'Dropout'):
    """
    for weight reusing

    see BIEEGConvNet
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'SpatialDropout3D':
        dropoutType = SpatialDropout3D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D, '
                         'SpatialDropout3D or Dropout, passed as a string.')
    # Learn from raw EEG signals
    l_m = []
    input_s = Input(shape=(Chans, Samples, Colors), dtype=dtype)
    for i in range(Colors):
        input = Input(shape=(Chans, Samples, 1), dtype=dtype)
        s = Conv2D(F1, (1, kernLength), padding = 'same', use_bias = False)(input)
        s = BatchNormalization(axis = -1)(s)
        s = DepthwiseConv2D((Chans, 1), use_bias = False, depth_multiplier = D,
                            depthwise_constraint = max_norm(1.))(s)
        s = BatchNormalization(axis = -1)(s)
        s = Activation('elu')(s)
        s = AveragePooling2D((1, 4))(s)
        s = dropoutType(dropoutRate)(s)
        s = SeparableConv2D(F2, (1, 16), padding = 'same', use_bias = False)(s)
        s = BatchNormalization(axis = -1)(s)
        s = Activation('elu')(s)
        s = AveragePooling2D((1, 8))(s)
        s = dropoutType(dropoutRate)(s)
        model = Model(inputs=input,outputs=s)
        l_m.append(model)
    l_s = []
    for i in range(Colors):
        l_s.append(l_m[i](Lambda(lambda s:s[:,:,:,i:i+1])(input_s)))
    con = Concatenate(axis=-1)(l_s)
    #l = []
    #for i in range(2*D):
    #    for j in range(Colors):
    #        l.append(Lambda(lambda s:s[:,:,:,i+2*D*j:i+2*D*j+1])(con))
    #con = Concatenate(axis=-1)(l)
    #con = Conv2D(F1, (kernLength, 1), padding = 'same', use_bias = False, data_format = 'channels_first')(con)
    #con = BatchNormalization(axis = 1)(con)
    #con = DepthwiseConv2D((1, Colors), use_bias = False, depth_multiplier = D,
    #                      depthwise_constraint = max_norm(1.), data_format = 'channels_first')(con)
    #con = BatchNormalization(axis = 1)(con)
    #con = Activation('elu')(con)
    #con = AveragePooling2D((1, F2), data_format = 'channels_first')(con)
    #con = dropoutType(dropoutRate)(con)
    con = Conv2D(F2, (1, 1), padding = 'same', use_bias = False)(con)
    con = BatchNormalization(axis = -1)(con)
    con = Activation('elu')(con)
    #con = AveragePooling2D((1, 2))(con)
    #con = AveragePooling2D((1, 5), data_format = 'channels_first')(con)
    con = dropoutType(dropoutRate)(con)
    flatten = Flatten()(con)

    return Model(inputs=input_s,outputs=flatten)


def graphEEGConvModel(Colors, Samples, H, W, dropoutRate = 0.5,
                    kernLength = 64, F3 = 32, F4 = 40, F5 = 24,
                    dtype = tf.float32, dropoutType = 'Dropout'):
    """
    for weight reusing

    see BIEEGConvNet
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'SpatialDropout3D':
        dropoutType = SpatialDropout3D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D, '
                         'SpatialDropout3D or Dropout, passed as a string.')
    # Learn from EEG graphs
    input_g = Input(shape=(Colors, H, W, Samples), dtype=dtype)
    g = Conv3D(F3, (3, 3, kernLength), padding = 'same', use_bias = False)(input_g)
    g = BatchNormalization(axis = 1)(g)
    g = Activation('relu')(g)
    g = AveragePooling3D((3, 3, 4))(g)
    g = dropoutType(dropoutRate)(g)
    g = Conv3D(F4, (3, 3, kernLength), padding = 'same', use_bias = False)(g)
    g = BatchNormalization(axis = 1)(g)
    g = Activation('relu')(g)
    g = AveragePooling3D((3, 3, 8))(g)
    g = dropoutType(dropoutRate)(g)
    g = Conv3D(F5, (3, 3, kernLength), padding = 'same', use_bias = False)(g)
    g = BatchNormalization(axis = 1)(g)
    g = Activation('relu')(g)
    g = AveragePooling3D((3, 3, 16))(g)
    g = dropoutType(dropoutRate)(g)
    flatten = Flatten()(g)

    return Model(inputs=input_g,outputs=flatten)


def rawEEGConvNet(n_classes, model, Colors=1, Chans=22, 
                  Samples=1000, norm_rate=0.25, dtype=tf.float32):
    # Learn from raw EEG signals
    input_s = Input(shape=(Chans, Samples, Colors), dtype=dtype)
    s = model(input_s)
    _output_s = Dense(n_classes, activation='softmax', kernel_constraint = max_norm(norm_rate))(s)

    return Model(inputs=input_s,outputs=_output_s)


def graphEEGConvNet(n_classes, model, Colors=16, H=16, 
                    W=16, Samples=1000, norm_rate=0.25, dtype=tf.float32):
    # Learn from EEG graphs
    input_g = Input(shape=(Samples, H, W, Colors), dtype=dtype)
    g = model(input_g)
    _output_g = Dense(n_classes, activation='softmax', kernel_constraint = max_norm(norm_rate))(g)

    return Model(inputs=input_g,outputs=_output_g)


def BiInputsEEGConvNet(n_classes, model_s, model_g, Chans=22, 
                 Samples=1000, Colors=16, H=16, W=16,
                 norm_rate = 0.25, dtype = tf.float32):
    """
    data_format: NCHW for 4D data
                 NCHWT for 5D data

    Inputs:
      n_classes       : int, number of classes to classify
      Chans, Samples  : int, number of channels and time points in the EEG data
      Colors, H, W    : int, filter numbers of filter-bank, height and width of the EEG graph
      norm_rate       : float, number of max_norm rate
      dtype           : object, type of data, default tf.float32

    Output:
      model           : BIEEGConvNet keras model
    """

    # Learn from raw EEG signals
    _input_s = Input(shape=(Chans, Samples, Colors), dtype=dtype)
    _s = model_s(_input_s)

    # Learn from EEG graphs
    _input_g = Input(shape=(Samples, H, W, Colors), dtype=dtype)
    _g = model_g(_input_g)

    # Merge both inputs and make predictions
    _x = Concatenate(axis=-1)([_s,_g])
    _output = Dense(n_classes, activation='softmax', kernel_constraint = max_norm(norm_rate))(_x)

    return Model(inputs=[_input_s,_input_g],outputs=_output)


def _old_BIEEGConvNet(n_classes, Chans, Samples, Colors, H, W,
                dropoutRate = 0.5, kernLength = 64, F1 = 8,
                D = 2, F2 = 16, F3 = 32, F4 = 40, F5 = 24,
                norm_rate = 0.25, dtype = tf.float32,
                dropoutType = 'Dropout'):
    """
    data_format: NCHW for 4D data
                 NCHWT for 5D data

    Inputs:
      n_classes       : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      Colors, H, W    : number of colors, height and width of the EEG graph
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn.
                        Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      F3, F4, F5      : number of graph filters (F3), number of graph filters (F4)
                        and number of graph filters (F5) to learn.
                        Default: F3 = 32, F4 = 40, F5 = 24.
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    Output:
      model           : BIEEGConvNet keras model
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'SpatialDropout3D':
        dropoutType = SpatialDropout3D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D, '
                         'SpatialDropout3D or Dropout, passed as a string.')
    # Learn from raw EEG signals
    input_s = Input(shape=(Colors, Chans, Samples), dtype=dtype)
    Ms = rawEEGConvModel(Chans=Chans, Samples=Samples,
                        dropoutRate=dropoutRate, kernLength=kernLength,
                        F1=F1, D=D, F2=F2, dtype=dtype,
                        dropoutType=dropoutType)
    s = Ms(input_s)

    # Learn from EEG graphs
    input_g = Input(shape=(Colors, Samples, H, W), dtype=dtype)
    Mg = graphEEGConvModel(Colors=Colors, Samples=Samples, H=H, W=W,
                            dropoutRate=dropoutRate, kernLength=kernLength,
                            F3=F3, F4=F4, F5=F5, dtype=dtype,
                            dropoutType=dropoutType)
    g = Mg(input_g)

    # Merge both inputs and make predictions
    x = Concatenate(axis=-1)([s,g])
    x = Dense(n_classes*10, activation='relu', kernel_constraint = max_norm(norm_rate))(x)
    predictions = Dense(n_classes, activation='softmax', kernel_constraint = max_norm(norm_rate))(x)

    return Model(inputs=[input_s,input_g],outputs=predictions)


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 4, F2 = 32, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = -1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = -1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16), padding = 'same', 
                                   use_bias = False)(block1)
    block2       = BatchNormalization(axis = -1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input(shape=(Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(1, Chans, Samples),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main   = Input(shape=(Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


if __name__=='__main__':
    print(tf.keras.__version__)