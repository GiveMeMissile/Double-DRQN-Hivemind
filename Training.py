import time
import utils
import pandas as pd
import Config as init
from DDRQN import AIHivemindManager
from DataManager import SequentialTrainingData
from RewardCalc import RewardCalculator

# Varaibles for this specific training session.
EPOCHS = 50
BATCH = 64
SEQUENCE_LENGTH = 200
BATCHES_PER_EPOCH = 50
NUM_LAYERS = init.NUM_LAYERS
HIDDEN_SIZE = init.HIDDEN_SIZE + 3
lr = 0.00001
data_number = 1
data_idx = 0

# Reward Variables:
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


def train(ai_manager, data_manager):

    total_time = 0
    beginning_time = time.time()
    avg_loss = 0

    for batch_num in range(BATCHES_PER_EPOCH):
        # Get sequence data
        sequence = data_manager.get_sample_sequence(
            BATCH,
            sequence_length = SEQUENCE_LENGTH
        )

        # Check sequence data to ensure that it is good
        if sequence is None:
            continue
        
        # Train AI and get loss
        loss = ai_manager.sequence_training(sequence)
        avg_loss += loss

        # Sync models
        if batch_num % 2 == 0:
            ai_manager.sync_models()
    
    # Calculate variabes to be displayed
    avg_loss = avg_loss / BATCHES_PER_EPOCH
    ending_time = time.time()
    total_time = round(((ending_time-beginning_time)), 3)
    return total_time, loss


def prepare_ai(metadata):
    num_ais = metadata["Ais"][data_idx]
    num_glues = metadata["Glues"][data_idx]
    num_frames = metadata["Saved Frames"][data_idx]
    sequence_length = 4 + (num_ais * 13) + (num_glues * 2)

    ai_manager = AIHivemindManager(
        num_ais, 
        None, # Uneeded parameters are set to none.
        None, 
        NUM_LAYERS, 
        HIDDEN_SIZE, 
        init.OUTPUT_SIZE, 
        lr, 
        None, 
        None,
        num_frames,
        sequence_length
    )
    return ai_manager

def correct_rewards():
    if not (init.MIN_REWARDS == MIN_REWARDS):
        return False
    if not (init.MAX_REWARDS == MAX_REWARDS):
        return False
    if not (init.VELOCITY_REWARD_MULTIPLIER == VELOCITY_REWARD_MULTIPLIER):
        return False
    if not (init.DISTANCE_REWARD_MULTIPLIER == DISTANCE_REWARD_MULTIPLIER):
        return False
    if not (init.PLAYER_CONTACT == PLAYER_CONTACT):
        return False
    if not (init.GLUE_CONTACT == GLUE_CONTACT):
        return False
    if not (init.WALL_CONTACT == WALL_CONTACT):
        return False
    if not (init.MOVING_TOWARDS_PLAYER == MOVING_TOWARDS_PLAYER):
        return False
    if not (init.NO_MOVEMENT == NO_MOVEMENT):
        return False
    if not (init.AI_COLLISION == AI_COLLISION):
        return False
    return True

def modify_rewards(reward_calculator, data, metadata):
    print("Preparing Data...")
    for i in range(len(data)):
        for j in range(len(data)):
            tensor = data[i][0][j]
            p, a, g = reward_calculator.denormalize_tensor_data(tensor[2], metadata, data_idx, tensor[0])
            data[i][0][j][3] = reward_calculator.calculate_rewards(g, a, p)
    print("Data Has been prepared.")
    return data

if __name__ == "__main__":

    data_manager = SequentialTrainingData(
        init.TRAINING_DATA_FOLDER,
        init.TRAINING_DATA_INFO,
        max_episodes=200
    )
    progress_data = {
        "Epoch": [],
        "Loss": [],
        "Time": []
    }
    if data_idx == -1:
        data_idx = utils.get_idx_from_number(data_manager.get_metadata(), data_number, "Data Number")
    ai_manager = prepare_ai(data_manager.get_metadata())
    print("Loading data...")
    data_manager.load_data(number=data_number)
    print("Data has been loaded.")

    if not correct_rewards():
        reward_calculator = RewardCalculator(
            MIN_REWARDS,
            MAX_REWARDS,
            VELOCITY_REWARD_MULTIPLIER,
            DISTANCE_REWARD_MULTIPLIER,
            PLAYER_CONTACT,
            GLUE_CONTACT,
            WALL_CONTACT,
            MOVING_TOWARDS_PLAYER,
            NO_MOVEMENT,
            AI_COLLISION
        )
        data_manager.episodes = modify_rewards(reward_calculator, data_manager.episodes, data_manager.get_metadata())


    print("Training will now commence")
    for epoch in range(EPOCHS):
        total_time, loss = train(ai_manager, data_manager)
        print(f"Epoch: {epoch+1} | Time: {total_time} | Loss: {loss}")
        progress_data["Epoch"].append(epoch+1)
        progress_data["Loss"].append(loss)
        progress_data["Time"].append(total_time)
    print("Training Has been Complete")
    ai_manager.save_model(file=init.INFO_FILE, save_folder=init.SAVE_FOLDER)


