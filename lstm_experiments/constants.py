DATA_FILEPATH = 'data/japan_earthquakes.csv'
MODEL_SAVE_DIR = 'models'
VISUALS_DIR = 'figures'

EPOCHS = 100
BATCH_SIZE = 128
TEST_SPLIT = 0.2
LEARNING_RATE = 0.00005
PATIENCE = 20

MODEL_DIMS = {
    'hidden_size': 1500,
    'dense_size_1': 250,
    'dense_size_2': 15,
    'output_size': 1,
}