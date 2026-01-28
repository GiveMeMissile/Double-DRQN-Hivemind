#######################################################################################################
# File Name:
#   utils
#
# Purpose:
#   Contains functions used throughout the program.
#   There isn't much else to it. 
#
# Used:
#   1 - Used by main to help the game run.
#   2 - Used by DataManger, specifically the get_lowest() function.
#   3 - Used by DDRQN, also the get_lowest() function.
#######################################################################################################

import pygame
import torch
import os
import json
import Config as init


def kill_model(model_number=None):
    # Used to remove an saved Model file and the JSON metadata attached to it
    idx = -1
    with open(init.INFO_FILE) as f:
        model_info = json.load(f)

    if model_number is None:
        if len(model_info["Model"]) == 0:
            return None
        
        model_file = None

        for i in range(len(model_info["Model"])):
            ais = model_info["Ais"][i] == init.NUM_AI_OBJECTS
            glues = model_info["Glues"][i] == init.GLUES
            hidden = model_info["Hidden"][i] == init.HIDDEN_SIZE
            frames = model_info["Frames"][i] == init.NUM_SAVED_FRAMES
            layers = model_info["Layers"][i] == init.NUM_LAYERS

            if ais and glues and hidden and frames and layers:
                idx = i
                model_number = model_info["Model"][idx]
                print(f"Now Deleting Model {model_number}...")
                break
    else:
        idx = get_idx_from_number(model_info, model_number, "Model")
                


    if model_number is None or idx == -1:
        print(f"Model number {model_number} does not exist, please try a model number which actually exists.")
        return -1

    model_file = init.SAVE_FOLDER + '/model_' + str(model_number) + ".pth"

    if os.path.exists(model_file):
        os.remove(model_file)
    else:
        print(f"Was unable to find model {model_number}\n")

    model_info["Model"].pop(idx)
    model_info["Ais"].pop(idx)
    model_info["Glues"].pop(idx)
    model_info["Hidden"].pop(idx)
    model_info["Frames"].pop(idx)
    model_info["Layers"].pop(idx)
    model_info["Epsilon"].pop(idx)
    with open(init.INFO_FILE, 'w') as f:
        json.dump(model_info, f)
        print("The model had been deleted successfully.\n")
    return model_number

def destroy_data(data_number):

    info_data = None
    with open(init.TRAINING_DATA_INFO, "r") as f:
        info_data = json.load(f)

    idx = -1
    idx = get_idx_from_number(info_data, data_number, "Data Number")

    if idx == -1:
        print(f"Error: Data number {data_number} does not exist...")
        return False
    
    filename = init.TRAINING_DATA_FOLDER + "/training_data_" + str(data_number) + ".pkl"
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("Error: File cannot be found")
        return False
    
    info_data["Data Number"].pop(idx)
    info_data["Glues"].pop(idx)
    info_data["Ais"].pop(idx)
    info_data["Saved Frames"].pop(idx)
    info_data["Window X"].pop(idx)
    info_data["Window Y"].pop(idx)
    info_data["Episodes"].pop(idx)
    with open(init.TRAINING_DATA_INFO, 'w') as f:
        json.dump(info_data, f)

    return True



def check_for_folder():
    # Creates AI save directory and JSON file if one is not present.

    # Model saving
    if not os.path.isdir(init.SAVE_FOLDER):
        os.mkdir(init.SAVE_FOLDER)
    if not os.path.isfile(init.INFO_FILE):
        open(init.INFO_FILE, "x")
        with open(init.INFO_FILE, "w") as f:
            json.dump(init.AI_SAVE_DATA, f)

    # Progress data
    if not os.path.isdir(init.DATA_FOLDER):
        os.mkdir(init.DATA_FOLDER)

    # Training data
    if not os.path.isdir(init.TRAINING_DATA_FOLDER):
        os.mkdir(init.TRAINING_DATA_FOLDER)
    if not os.path.isfile(init.TRAINING_DATA_INFO):
        open(init.TRAINING_DATA_INFO, "w")
        with open(init.TRAINING_DATA_INFO, "w") as f:
            json.dump(init.TRAINING_SAVE_DATA, f)


def display_training_text(text, window):
    # Displays the inputted text when training an AI

    window.fill(init.BLACK)
    top_text = init.training_font.render("Training AI:", True, init.WHITE)
    bottom_text = init.font.render(text, True, init.WHITE)

    top_rect = top_text.get_rect(center=(init.WINDOW_X/2, init.WINDOW_Y/2))
    bottom_rect = bottom_text.get_rect(center=(init.WINDOW_X/2, init.WINDOW_Y/2 + top_rect.height))

    window.blit(top_text, top_rect)
    window.blit(bottom_text, bottom_rect)
    pygame.display.flip()


def post_game_training(hivemind_manager, sequential_data, episodes, window):
    # Trains the AI after an episode ends when training=True

    # Determines training amount based off of the amount of data collected.
    if len(sequential_data) < 5:
        batch_size = init.TRAINING_BATCH_SIZE // 2**3
        num_batches = init.NUM_BATCHES // 4
    elif len(sequential_data) < 25:
        batch_size = init.TRAINING_BATCH_SIZE // 2**2
        num_batches = init.NUM_BATCHES // 2
    elif len(sequential_data) < 50:
        batch_size = init.TRAINING_BATCH_SIZE // 2
        num_batches = (3 * init.NUM_BATCHES) // 4
    else:
        batch_size = init.TRAINING_BATCH_SIZE
        num_batches = init.NUM_BATCHES

    end = False
    total_time = 0
    bottom_text = "Beginning post game training..."
    print("\n" + bottom_text)
    display_training_text(bottom_text, window)

    batch_count = 0
    beginning_time = pygame.time.get_ticks()
    epoch_loss = 0
    time = 0
    avg_loss = torch.tensor([0])

    for batch_num in range(num_batches):
        print(f"Batch {batch_num + 1}/{num_batches} finished in {time} seconds")
        text = "Batch: " + str(batch_num) + "/" + str(num_batches) + " | Loss: " + str(round(avg_loss.item(), 5)) + " | Total Time: " + str(total_time)
        display_training_text(text, window)
        beginning_time = pygame.time.get_ticks()
        epoch_loss = 0
        # Get sequence data
        sequence = sequential_data.get_sample_sequence(
            batch_size,
            sequence_length = init.TRAINING_SEQUENCE_LENGTH
        )

        # Check sequence data to ensure that it is good
        if sequence is None:
            continue
        
        # Train AI and get loss
        loss = hivemind_manager.sequence_training(sequence)
        epoch_loss += loss
        batch_count += 1

        # Sync models
        if batch_count % 2 == 0:
            hivemind_manager.sync_models()

        # Check for QUIT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
                break
        if end:
            break
        
        # Calculate variabes to be displayed
        avg_loss = epoch_loss
        ending_time = pygame.time.get_ticks()
        time = round(((ending_time-beginning_time)/1000), 3)
        total_time += time
        total_time = round(total_time, 3)

    return end


def game_end(ai_manager, player, progress_tracker, time, data_manager, end, episodes, window):

    data_manager.episode_end()

    if init.training and not end:
        end = post_game_training(ai_manager, data_manager, episodes, window)

    if init.EPSILON_ACTIVE and init.training:
        # Decay Epsilon
        new_epsilon = (ai_manager.epsilon - init.INITIAL_EPSILON/init.EPSILON_DECAY)
        if new_epsilon < 0:
            new_epsilon = 0
    else:
        new_epsilon = ai_manager.epsilon

    ai_manager.epsilon = new_epsilon

    new_previous_time = pygame.time.get_ticks()

    # Save Progress to AI
    rewards = ai_manager.total_rewards/ai_manager.reward_count
    try:
        loss = ai_manager.total_loss/ai_manager.loss_count
    except ZeroDivisionError:
        loss = 0
    actions = 0
    epsilon = ai_manager.epsilon
    health = player.health
    spread = 0
    time = round((init.TIME_LIMIT - time)/1000)
    for ai in ai_manager.ai_list:
        actions += ai.total_actions/ai.action_count
        spread += get_spread(ai.action_distribution, 0, 8)
        ai.action_distribution.clear()

    actions /= len(ai_manager.ai_list)
    spread /= len(ai_manager.ai_list)
    q_value_max = ai_manager.q_value_max/ai_manager.q_value_count
    q_value_average = ai_manager.q_value_average/ai_manager.q_value_count
    q_value_min = ai_manager.q_value_min/ai_manager.q_value_count
    if init.training:
        q_value_train = ai_manager.q_value_train_avg/ai_manager.q_value_train_count
    else:
        q_value_train = 0

    progress_tracker.model_number = ai_manager.model_number
    progress_tracker.append(time, "Time")
    progress_tracker.append(rewards, "Rewards")
    progress_tracker.append(loss, "Loss")
    progress_tracker.append(epsilon, "Epsilon")
    progress_tracker.append(health, "Health")
    progress_tracker.append(actions, "Actions")
    progress_tracker.append(spread, "Action Spread")
    progress_tracker.append(q_value_max, "Q Values Max")
    progress_tracker.append(q_value_average, "Q Values Average")
    progress_tracker.append(q_value_min, "Q Values Min")
    progress_tracker.append(q_value_train, "Q Values Train")
    
    return end, new_previous_time


def get_spread(number_list, min_num, max_num):

    if not number_list:
        return 0

    num_items = len(number_list)
    number_of_each = []
    for i in range(min_num, max_num+1):
        number_of_each.append(number_list.count(i))
        # print(f"Count {i}: {number_of_each[i-min_num]}") # debug line
    number_of_calculations = 0
    spread = 0
    for i, (count) in enumerate(number_of_each):
        for j in range(i):
            if i == j:
                continue
            number_of_calculations += 1
            reciving_count = number_of_each[j]
            if (count == 0 and reciving_count == 0):
                continue
            spread += 1-(abs(count-reciving_count)/num_items)

    return spread/number_of_calculations


def draw_game(player, glues, time, ai_manager, episodes, window):
    # Draws the game on the window. Quite self explanatory.
    window.fill(init.BLACK)

    player.display()

    for ai in ai_manager.ai_list:
        ai.display()

    for glue in glues:
        glue.display()

    try: 
        text = init.font.render(f"Health: {player.health}  |  Time Remaining: {round((init.TIME_LIMIT - time)/1000)}  |  Episode: {episodes}/{init.MAX_EPISODES}", True, init.WHITE)
    except Exception:
        text = init.font.render("Health: 0", True, init.WHITE)
    window.blit(text, (10, 10))
    pygame.display.flip()


def get_lowest(data):
    # Takes in a list of ints and checks for the lowest number which does not exist, (used for file saving).

    if data == []:
        return 1
    data.sort()
    if data[0] > 1:
        return 1
    for i in range(len(data)-1):
        num = data[i]
        upper_num = data[i+1]
        if upper_num - num > 1:
            return num+1
    return max(data) + 1

def copy_raw_tensor_data(tensor): # 1d tensor
    tl = []
    for i in range(tensor.size()[0]):
        tl.append(tensor[i].item())
    return torch.tensor(tl)

def get_idx_from_number(data, number, name):
    # Takes in a number from either ai or training metadata and uses it to locate the idx.

    for i, part in enumerate(data[name]):
        if part == number:
            return i
        
    return -1
