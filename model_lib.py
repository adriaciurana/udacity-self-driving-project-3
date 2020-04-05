import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

IM_CROP = [
    (50, 140), 
    (0, 320)
]
IM_SHAPE = [
    IM_CROP[0][1] - IM_CROP[0][0], 
    IM_CROP[1][1] - IM_CROP[1][0]
]

def get_model(IM_SHAPE):
    base_model = MobileNet(weights='imagenet', include_top=False)

    #input_im = Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3))
    input_im = Input(shape=(IM_SHAPE[0], IM_SHAPE[1], 3))
    x = base_model(input_im)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x_steer = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_im, outputs=x_steer)

def predict(model, images):
    image_array = image_array[IM_CROP[0][0]:IM_CROP[0][1], IM_CROP[1][0]:IM_CROP[1][1]]
    steering_angle = float(model.predict(preprocess_input(image_array[None, :, :, :]))[0])
    steering_angle = (2 * steering_angle) - 1

    return steering_angle