import numpy as np
import tensorflow as tf
import scipy
import sklearn
import matplotlib

import scipy.sparse as spr

import tensorflow.python.keras as keras
from tensorflow.python.keras import optimizers as ops
from tensorflow.python.keras.layers import Dense, Dropout, Input

import data_handler
from model import MyModel

# ============================= GLOBAL VARIABLES =============================
# !!! IMPORTANT !!!
# Ensure valid file path to facebook.npz here
# Should be in the resources folder, in the same directory as this file
FILE_PATH = r"./facebook_large/facebook.npz"

# Plotting Variables
PLOT_ACCURACY = True  # Set whether or not you want to plot accuracy
PLOT_TSNE = False  # Set whether or not you want to plot accuracy

# Model Variables
EPOCHS = 50  # Set the number of epochs over which the Model should train
LEARNING_RATE = 0.01  # Set the Model learning rate
DROPOUT_RATE = 0.4

# Data Splitting Variables
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.10
VAL_SPLIT = 0.10
# ============================================================================


def main():
    print("=================================")
    print("======== STARTING PROGRAM =======")
    print("=================================")

    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)
    print("SciPy version:", scipy.__version__)
    print("SkLearn version:", sklearn.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Keras version:", keras.__version__)

    # ================================= DATA =================================
    # File Path
    data = np.load(FILE_PATH)

    # Parse Data
    data_split = (TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT)
    a_bar, feats, labels, num_nodes = data_handler.parse_data(data)

    # Split Train/Test/Validate
    train_mask, test_mask, val_mask = data_handler.generate_masks(data_split,
                                                                  num_nodes)
    validation_data = ([feats, a_bar], labels, val_mask)

    # ================================ MODEL ================================
    # Create Model
    gcn_model: keras.Model = MyModel(input_shape=tf.Tensor.get_shape(feats))

    # Compile Model
    gcn_model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Train Model
    history = gcn_model.fit(x=[feats, a_bar],
                            y=labels,
                            sample_weight=train_mask,
                            batch_size=22470,
                            epochs=EPOCHS,
                            shuffle=False,
                            validation_data=validation_data
                            )

    # Test Model
    gcn_model.evaluate(x=[feats, a_bar],
                       y=labels,
                       batch_size=22470,
                       sample_weight=test_mask,
                       )

    # Plot Accuracy
    if PLOT_ACCURACY:
        data_handler.generate_accuracy_plot(history)

    # Plot TSNE
    if PLOT_TSNE:
        data_handler.generate_tsne_plot(labels, feats, "Data")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

