# Imports
from  keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

def vgg16_finetuned():

	image_size = 224

	# Load the VGG model
	vgg_conv = VGG16(weights='imagenet',
					 include_top=False,
					 input_shape=(image_size, image_size, 3))

	# Freeze all the layers except the last 4 layers
	for layer in vgg_conv.layers[:-4]:
		layer.trainable = False

	# Check the trainable status of the individual layers
	# for layer in vgg_conv.layers:
	#     print(layer, layer.trainable)

	# Create a Sequential model
	model = models.Sequential()

	# Add the vgg convolutional base model to the Sequential model
	model.add(vgg_conv)

	# Add new layers
	model.add(layers.Flatten())
	model.add(layers.Dense(1024, activation='relu'))
	model.add(layers.Dropout(0.8))
	model.add(layers.Dense(2, activation='softmax'))

	return model
