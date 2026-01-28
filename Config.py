import torch
import pygame

# Game constants
MAX_EPISODES = 150
WINDOW_X, WINDOW_Y = 1500, 750
TIME_LIMIT = 60000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Glue constants
GLUE_DIM = 75
GLUES = 5
GLUE_MOVEMENT_TIME = 5000

# Other object constants (player + AI objects)
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 15
DAMAGE_COOLDOWN = 1000
PLAYER_DIM = 50
RANDOM_MOVE = True
REMOVE_OBJ_TIME = 0
PLAYER_TIME_MAX, PLAYER_TIME_MIN = 100, 25

# AI Constants:

# Object: 
COLLISION_TIMER = 500
NUM_AI_OBJECTS = 1

# Hyperparameters:
NUM_LAYERS = 4
HIDDEN_SIZE = 64*8
NUM_SAVED_FRAMES = 20
LEARNING_RATE = 0.000005
SEQUENCE_LENGTH = 4 + ((13) * NUM_AI_OBJECTS) + (2 * GLUES) # 4 for player, 13 for each AI, 2 for each glue.
INPUT_SHAPE = (NUM_SAVED_FRAMES, SEQUENCE_LENGTH)
DISCOUNT_FACTOR = 0.95
OUTPUT_SIZE = 9

# Training:
SYNC_MODEL = 200
GAME_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = 32
TRAINING_SYNC = 10
TRAINING_SEQUENCE_LENGTH = 100
NUM_BATCHES = 50
EPSILON_ACTIVE = True # Determines if epsilon is active
BIASED_EPSILON = True
EPSILON_TIME_MAX, EPSILON_TIME_MIN = 200, 50
TRAINING_INCREMENT = 100 # How many episodes/games the ai plays until values return to the original
EPSILON_DECAY = 100
INITIAL_EPSILON = 1

# Data Config
DATA_COLLECTION = True
PROGRESS_DATA_SAVE = False
LOAD_PREVIOUS_DATA = False

# Data Saving Format
AI_SAVE_DATA = {
    "Model": [],
    "Ais": [],
    "Glues": [],
    "Hidden": [],
    "Frames": [],
    "Layers": [],
    "Epsilon": []
}
TRAINING_SAVE_DATA = {
    "Data Number": [],
    "Ais": [],
    "Glues": [],
    "Saved Frames": [],
    "Window X": [],
    "Window Y": [],
    "Episodes": []
}

# Files
SAVE_FOLDER = "RL_LSTM_Models"
INFO_FILE = SAVE_FOLDER + "/" +"model_info.json"
DATA_FOLDER = "RL_LSTM_Progress_Data"
TRAINING_DATA_FOLDER = "LSTM_Training_Data"
TRAINING_DATA_INFO = TRAINING_DATA_FOLDER + "/" + "training_info.json"


# AI Reward values, reward values should in the interval [-10, 10]. If rewards exceed the interval, they will just be the max/min of the interval.
MIN_REWARDS = -10
MAX_REWARDS = 10
VELOCITY_REWARD_MULTIPLIER = 3
DISTANCE_REWARD_MULTIPLIER = 2
PLAYER_CONTACT = 10
GLUE_CONTACT = -5
WALL_CONTACT = -3
MOVING_TOWARDS_PLAYER = 2
NO_MOVEMENT = -2
AI_COLLISION = -5

# Other important stuff
iteration = 51  # Used for data saving and testing purposes.
training = False  # If AI is going to be actively training (if true then activiates curriculum, data saving, and post game training)
delete_model_file = False # If True then if a model file exists for the current variables, it gets deleated and replaced by a new model.
curriculum = False # Determines if curriculum is active or not.
device = "cuda" if torch.cuda.is_available() else "cpu"  # Device agnostic code ig
previous_time = 0  # Used for time calculation for each game/episode.
pygame.font.init()  # I am so gooberish ig idk please turn me into a fish.
font = pygame.font.SysFont("New Roman", 30)  # Its font idk what to say
training_font = pygame.font.SysFont("New Roman", 50)  # font, but for training (: