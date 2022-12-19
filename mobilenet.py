import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, GlobalAveragePooling2D, BatchNormalization, Activation, Flatten, Reshape, Dense, Dropout

def create_model(alpha=1, imgsize=224, num_classes=1000, dropout_rate=1e-3, shallow=False):
    """
    Creates the basic structure of MobileNet.
    Each pointwise convolution layer is followed by batchnorm, ReLU and Dropout; each depthwise convolution layer is followed by batchnorm and ReLU.
    Arguments: 
    alpha: width multiplier. alpha is a number between 0 and 1. alpha = 1 is the baseline MobileNet and α < 1 are reduced MobileNets.
    imgsize: shape of input images. For example, if input image has a shape of (224, 224, 3), imgsize should be set to 224.
    num_classes: specifies the number of classes in the training data.
    dropout_rate: defines the dropout rate of each dropout layer.
    shallow: a boolean to determine whether a shallower MobileNet should be built.
    """
    model = tf.keras.Sequential() 
    model.add(Conv2D(filters=32*alpha, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(imgsize,imgsize,3)))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=64*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=128*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=128*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=256*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=256*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=512*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))
    
    if shallow==False:
        for i in range(5):
            model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU(6.))

            model.add(Conv2D(filters=512*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU(6.))
            model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=1024*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    model.add(Conv2D(filters=1024*alpha, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Dropout(dropout_rate))

    #model.add(AveragePooling2D(pool_size=(7,7), strides=(1,1), data_format='channels_first'))
    model.add(GlobalAveragePooling2D())
    model.add(Reshape((1, 1, int(1024 * alpha))))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(num_classes, kernel_size=(1,1), padding='same'))
    model.add(Reshape((num_classes,)))
    model.add(Activation('softmax'))
    return model
