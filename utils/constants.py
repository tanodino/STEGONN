OUTPUT_FOLDER = './extended_output/starcane2_dordogne'
BASE_FILE_NAMES = ['x_train_', 'x_validation_', 'x_test_', 'y_train_', 'y_validation_', 'y_test_',
                   'id_train_', 'id_validation_', 'id_test_']
EXTENSION = '.npy'
N_BANDS = 6
BATCH_SIZE = 32
EPOCHS = 3000

""" REUNION - DATASET 
BASE_DIR_FOLDS = '../../Reunion-Dataset/fold_extended_crop '
REUNION_LABEL_NAMES = ["Sugar cane", "Pasture and fodder", "Market gardening", "Greenhouse crops or shadows", "Orchards",
                       "Wooded areas", "Moor and Savannah", "Rocks and bare soil", "Relief shadows", "Water", "Urbanized areas"]
REUNION_ALPHAS_FN = "reunion_alphas.txt"
OBJECTS_GT_FILE_NAME = '../../Reunion-Dataset/crop/slic_object_id_reunion_crop.npy'
GLOBAL_DATA_FILE_NAME = '../../Reunion-Dataset/crop/slic_data_reunion_crop.npy'
RAG_FILE_NAME = '../../Reunion-Dataset/crop/rag_slic_reunion'
N_TIMESTAMPS = 21
N_CLASSES = 11
MAX_LIMIT = 8
"""

""" DORDOGNE DATASET """
BASE_DIR_FOLDS = '../../Dordogne-Dataset/fold '
OBJECTS_GT_FILE_NAME = '../../Dordogne-Dataset/slic_object_id_dordogne.npy'
DORDOGNE_ALPHAS_FN = "dordogne_alphas.txt"
DORDOGNE_LABEL_NAMES = ["Built-up", "Crops", "Water", "Forest", "Moor", "Orchards", "Vines"]
GLOBAL_DATA_FILE_NAME = '../../Dordogne-Dataset/slic_data_dordogne.npy'
RAG_FILE_NAME = '../../Dordogne-Dataset/rag_segmentation_dordogne'
N_TIMESTAMPS = 23
N_CLASSES = 7
MAX_LIMIT = 10
