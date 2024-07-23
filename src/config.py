import os

# Define base paths and filenames for data and model artifacts within the project.
# BASE_DATA_PATH specifies the directory where the dataset is located.
BASE_DATA_PATH = './data/NIH_Dataset'

# MODEL_SAVE_PATH specifies the path where the best performing model weights will be saved.
MODEL_SAVE_PATH = 'model_weights/best_model.pth'

# GRAPHS_SAVE_PATH defines the path to save performance graphs, like loss and accuracy plots.
GRAPHS_SAVE_PATH = 'performance_graphs/plot'

# Constant for the name of the column in the CSV file that contains diagnostic labels.
FINDING_LABELS_COLUMN = 'Finding Labels'

# Path to the CSV file that contains metadata about the images, using os.path.join for cross-platform compatibility.
CSV_FILE = os.path.join(BASE_DATA_PATH, 'Data_Entry_2017.csv')

# IMAGE_PATH_PATTERN defines a pattern to locate all PNG images within the specified subdirectories under BASE_DATA_PATH.
# It uses wildcard characters to include all directories starting with 'images_' and ending in '.png'.
IMAGE_PATH_PATTERN = os.path.join(
    BASE_DATA_PATH, 'images_*', 'images', '*.png')

# Configuration for training: BATCH_SIZE is the number of images processed in one iteration of model training.
BATCH_SIZE = 32

# VAL_BATCH_SIZE is the number of images processed in one iteration during validation phase, ensuring consistent batch processing.
VAL_BATCH_SIZE = 32

# NUM_EPOCHS defines the total number of complete passes through the training dataset.
NUM_EPOCHS = 10
