# The folder structure should be organized by data class, followed by each slide's name within each folder.
# ex)- ./data/train/N/2020S1245215/patch_name
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
TEST_DIR = "./data/test"

CLASSES = ['D', 'M', 'N']
INITIAL_WSI_PER_CLASS = 5  # Number of WSIs in the initial training dataset
NUM_WSI_PER_GENERATION = 30 # Number of WSIs selected for each AL iteration 
NUM_PATCHES_PER_WSI = 40    # Number of patches selected for each WSI
BASE_DIR = "log/stomach/trial_" # # Location where results are recorded


RESTORE_EXTRACTION = True 

CHECK_NUM_PATCHES = True  # check whether the number of patches in a WSI is under 40
NUM_PATCH_FILTER = 80   # WSI filtering criteria - How many patches should there be at a minimum?
RANDOM_SEED = 2024


LOAD_WSI_CSV = False   # Should the selection of WSIs be consistent with previous choices as an on/off feature?
LOAD_WSI_LOCATION = ""

MODEL_NAME = 'resnet' #'vgg','resnet'
BATCH_SIZE = 128
LEARNING_RATE = 0.01  
NUM_EPOCHS = 50
CYCLES = 10 # AL round 