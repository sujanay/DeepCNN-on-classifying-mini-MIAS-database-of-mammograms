# Imports
import time
import argparse

# importing matplotlib.pyplot caused error
import matplotlib.pyplot as plt

from keras import optimizers
from models.vgg16_model import vgg16_finetuned
from datagenerator.datagenerator import train_datagenerator, validation_datagenerator
from IPython.display import display
from PIL import Image
from utils import *

def create_model():
    """
        This function creates the vgg16 model and prints the model
        summary to the console.
    """
    print("inside create model!!!")
    # Build vgg16 model
    model = vgg16_finetuned()

    # Display the model summary
    print('Summary of the model')
    model.summary()

    return model

def train(model, epochs=5, train_batchsize=20, val_batchsize=10):
    """
    :param model: vgg16 model
    :param epochs: number of epochs to iterate through the training images
    :param train_batchsize: batch size of training images
    :param val_batchsize: batch size of validation images
    :return: model: trained vgg16 model
    :return: history: history of training at each epochs
    """
    # Get the datagenerator for train and validation data
    print('Reading training and validation images...')
    train_generator = train_datagenerator(train_batchsize)
    validation_generator = validation_datagenerator(val_batchsize)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.adam(lr=1e-5),
                  metrics=['acc'])

    # Train the model
    train_start = time.clock()

    print('Started training...')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=2*train_generator.samples/train_generator.batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.samples/validation_generator.batch_size,
                                  verbose=1)

    train_finish = time.clock()
    train_time = train_finish - train_start

    # print the time taken to train the model
    print('Training completed in {0:.3f} minutes!'.format(train_time/60))

    # Save the model
    print('Saving the trained model...')
    model.save('trained_models/model.h5')

    print("Saved trained model in 'traned_models/ folder'!")

    return model, history

def show_graphs(history):
    """
    :param history: history of the model at each epoch during training
    """
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs1 = range(len(acc))

    plt.plot(epochs1, acc, 'b', label='Training acc')
    plt.plot(epochs1, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs1, loss, 'b', label='Training loss')
    plt.plot(epochs1, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('Training and validation loss')
    plt.show()

def Main():
    """
        This Main() function builds the model, trains the model with the product images
        and saves the trained model (model1.h5) in trained_models/ folder.
    # """
    # Build the model
    model = create_model()

    print("epochs, train_batchsize, val_batchsize", epochs, train_batchsize, val_batchsize)
    # Train the model with provided epochs and batch size, and save the model
    _, history = train(model,               # vgg16 model
                       epochs,              # number of epochs to iterate through
                       train_batchsize,     # training batch size
                       val_batchsize)       # validation batch size

    # Show the graphs
    show_graphs(history)

if __name__ == '__main__':
    Main()
