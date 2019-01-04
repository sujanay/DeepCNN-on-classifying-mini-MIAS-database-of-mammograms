from utils import *
from keras.preprocessing.image import ImageDataGenerator

# Train datagenerator
def train_datagenerator(train_batchsize=20):
	"""
		docs goes here!!!
	"""
	train_datagen = ImageDataGenerator(rescale=1. / 255,
					   rotation_range=20,
					   width_shift_range=0.2,
					   height_shift_range=0.2,
					   horizontal_flip=True,
					   fill_mode='nearest')

	# Data Generator for Training data
	train_generator = train_datagen.flow_from_directory(train_dir,
						  	    target_size=(image_size, image_size),
							    batch_size=train_batchsize,
							    class_mode='categorical')

	return train_generator

def validation_datagenerator(val_batchsize=10):
	"""
		docs goes here!!!
	"""
	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	# Data Generator for Validation data
	validation_generator = validation_datagen.flow_from_directory(validation_dir,
																  target_size=(image_size, image_size),
																  batch_size=val_batchsize,
																  class_mode='categorical',
																  shuffle=False)
	return validation_generator

def test_datagenerator():
	"""
		docs goes here!!!
	"""
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	# Data Generator for Test data
	test_generator = test_datagen.flow_from_directory(test_dir,
							  target_size=(image_size, image_size),
							  batch_size=test_batchsize,
							  class_mode='categorical',
							  shuffle=False)
	return test_dir, test_generator
