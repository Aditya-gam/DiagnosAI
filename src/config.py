import os

# Configuration parameters for the entire project
BASE_DATA_PATH = './data/NIH_Dataset'
MODEL_SAVE_PATH = 'model_weights/best_model.pth'
GRAPHS_SAVE_PATH = 'performance_graphs/plot'

FINDING_LABELS_COLUMN = 'Finding Labels'
CSV_FILE = os.path.join(BASE_DATA_PATH, 'Data_Entry_2017.csv')
IMAGE_PATH_PATTERN = os.path.join(BASE_DATA_PATH, 'images_*', 'images', '*.png')

BATCH_SIZE = 32
VAL_BATCH_SIZE = 256
NUM_EPOCHS = 25