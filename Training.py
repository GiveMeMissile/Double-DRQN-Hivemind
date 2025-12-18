import time
import pandas as pd
import Config as init
from DDRQN import AIHivemindManager
from DataManager import SequentialTrainingData

# Varaibles for this specific training session.
EPOCHS = 50
BATCH = 64
SEQUENCE_LENGTH = 200
BATCHES_PER_EPOCH = 50
NUM_LAYERS = init.NUM_LAYERS
HIDDEN_SIZE = init.HIDDEN_SIZE
lr = 0.000005
data_idx = 0


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
    ai_manager = prepare_ai(data_manager.get_metadata())
    print("Loading data...")
    data_manager.load_data(idx=data_idx)
    print("Data has been loaded.")
    print("Training will now commence")
    for epoch in range(EPOCHS):
        total_time, loss = train(ai_manager, data_manager)
        print(f"Epoch: {epoch+1} | Time: {total_time} | Loss: {loss}")
        progress_data["Epoch"].append(epoch+1)
        progress_data["Loss"].append(loss)
        progress_data["Time"].append(total_time)
    print("Training Has been Complete")
    ai_manager.save_model(file=init.INFO_FILE, save_folder=init.SAVE_FOLDER)


