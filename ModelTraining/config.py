"""
R-CNN
Main Configuration class.
Written by
"""

import numpy as np

class Config(object):
    """Main configuration class. For custom configurations, create a
    sub-class that inherits te main class and override properties
    that needs to be changed.
    """
    # Name the configurations. For example, 'Experiment 1', 'Experiment 2', 'Experiment 3', ...etc.
    NAME = None

    # Number OF GPUs to use
    GPU_COUNT = 1

    # Number of images to train with on each GPU.
    IMAGES_PER_GPU = 2

    # Input image resizing
    # Available resizing modes: none, square, pad64, crop
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 224
    IMAGE_MAX_DIM = 224

    # Minimum scaling ratio.
    IMAGE_MIN_SCALE = 0

    # Number of color channels per image.
    # RGB = 3, grayscale = 1, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 3

    # Maximum number of proposed regions considered for training
    MAX_REGIONS = 2000
    
    # source of trained weights
    WEIGHTS = None

    # Optimizer Learning rate
    LEARNING_RATE = 0.0001

    # Rotation range for Image Generator
    ROTATION_RANGE_IMAGEGEN = 90

    # Rotation range for Image Generator
    DTYPE_IMAGEGEN = 'uint8'

    # Metric monitored while training the model
    MONITOR = 'val_loss'

    # Save frequency of the model
    SAVE_FREQ = 'epoch'

    # Minimum change in the monitored quantity to qualify as an improvement during training
    MIN_DELTA = 0

    # Number of epochs with no improvement after which training will be stopped.
    PATIENCE = 100

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Number of epochs
    TOTAL_EPOCHS = 1000

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 2

    # Number of classification classes (including background)
    NUM_CLASSES = 2

    def __init__(self):
        """
        Set values of the config attributes.
        """

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM,
                                         self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM,
                                         self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

    def display(self):
        """
        Display Configuration values.
        """

        print("\nConfigurations:")
        for c in dir(self):
            if not c.startswith("__") and not callable(getattr(self, c)):
                print("{:30} {}".format(c, getattr(self, c)))
        print("\n")
