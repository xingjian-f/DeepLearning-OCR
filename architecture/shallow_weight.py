from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from util import categorical_accuracy_per_sequence


def build_shallow_weight(channels, width, height, output_size, nb_classes):
	# input
	inputs = Input(shape=(channels, height, width))
	# 1 conv
	conv1_1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu', 
		W_regularizer=l2(0.01))(inputs)
	bn1 = BatchNormalization(mode=0, axis=1)(conv1_1)
	pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(bn1)
	gn1 = GaussianNoise(0.5)(pool1)
	drop1 = SpatialDropout2D(0.5)(gn1)
	# 2 conv
	conv2_1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu',
		W_regularizer=l2(0.01))(gn1)
	bn2 = BatchNormalization(mode=0, axis=1)(conv2_1)
	pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(bn2)
	gn2 = GaussianNoise(0.5)(pool2)
	drop2 = SpatialDropout2D(0.5)(gn2)
	# 3 conv
	conv3_1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu',
		W_regularizer=l2(0.01))(drop2)
	bn3 = BatchNormalization(mode=0, axis=1)(conv3_1)
	pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(bn3)
	gn3 = GaussianNoise(0.5)(pool3)
	drop3 = SpatialDropout2D(0.5)(gn3)
	# 4 conv
	conv4_1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu',
		W_regularizer=l2(0.01))(gn3)
	bn4 = BatchNormalization(mode=0, axis=1)(conv4_1)
	pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(bn4)
	gn4 = GaussianNoise(0.5)(pool4)
	drop4 = SpatialDropout2D(0.5)(gn4)
	# flaten
	flat = Flatten()(gn4)
	# 1 dense
	dense1 = Dense(8, activation='relu', W_regularizer=l2(0.1))(flat)
	bn6 = BatchNormalization(mode=0, axis=1)(dense1)
	drop6 = Dropout(0.5)(bn6)
	# output
	out = []
	for i in range(output_size):
		out.append(Dense(nb_classes, activation='softmax')(bn6))
	if output_size > 1:
		merged_out = merge(out, mode='concat')
		shaped_out = Reshape((output_size, nb_classes))(merged_out)
		sample_weight_mode = 'temporal'
	else:
		shaped_out = out
		sample_weight_mode = None
	model = Model(input=[inputs], output=shaped_out)
	model.summary()
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=[categorical_accuracy_per_sequence],
				  sample_weight_mode = sample_weight_mode
				  )

	return model