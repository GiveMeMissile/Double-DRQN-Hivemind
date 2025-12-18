#######################################################################################################
# File Name:
#   DDRQN
#
# Purpose:
#   To create, manage, and save the AI model as well as 
#   managing its interactions with the enviorment.
#
# Used:
#   1 - Used by Training to load the model which will be trained.
#   2 - Used by main to save progress data and training data.
#   3 - Used by DataExtractor to analyze progress data and measure preformance.
#######################################################################################################

import torch
import pygame
import json
import math
import os
import Config as init
from torch import nn
from Objects import AI
from utils import get_lowest

class HivemindLSTM(nn.Module):
    # We will be using an LSTM model in this experiment. This LSTM will control the AI objects.
    # How many Networks can a Network Neural if a Network could Neural Networks?

    episode_h0 = None
    episode_c0 = None

    def __init__(self, num_layers, input_size, hidden_size, output_size, num_ais):
        super(HivemindLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_ais = num_ais
        self.num_layers = num_layers

        # LSTM used by the AI
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.shared_layer = nn.Linear(hidden_size, hidden_size)
        self.split_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, output_size)
            )
            for _ in range(num_ais)
        ])

    def forward(self, state, h0=None, c0=None):
        # Input tensor ideas: sequence = [aix, aiy, dx, dy, player_x, player_y, glue_x1, glue_y1, glue_x2, glue_y2, ...]
        # where dx and dy are the player's velocity, player_x and player_y are the player's position,
        # Input_Tensor = [BATCH, NUM_SAVED_FRAMES, sequence length]. Where there will be NUM_SAVED_FRAMES amount of sequances in the tensor.
        # The sequences include the current frame plus the last NUM_SAVED_FRAMES-1 frames.

        # Create Cells if they do not exist
        if (h0 is None):
            h0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(init.device)
        if (c0 is None):
            c0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(init.device)
        
        # Put the state through the LSTM
        out, (h0, c0) = self.lstm(state, (h0, c0))

        # Input the output from the LSTM to the shared layer.
        shared_features = torch.relu(self.shared_layer(out[:, -1, :]))

        # Use the split layers
        ai_outputs = []
        for i in range(self.num_ais):
            ai_q_values = self.split_layers[i](shared_features)
            ai_outputs.append(ai_q_values)

        ai_output = torch.stack(ai_outputs, dim=1)

        # Output tensor: [action 0, action 1, action 2, action 3, action 4, action 5, action 6, action 7, action 8]
        # Note: the actions are structured as a binary tensor which determine which direction the thing go. [up, down, left, right]
        # 0 = False, 1 = True
        # action 0 = [0, 0, 0, 0] Going nowhere (Like my life)
        # action 1 = [1, 0, 0, 0] Going up
        # action 2 = [0, 1, 0, 0] Going down
        # action 3 = [0, 0, 1, 0] Going left
        # action 4 = [0, 0, 0, 1] Going right
        # action 5 = [1, 0, 1, 0] Going Up + Left
        # action 6 = [1, 0, 0, 1] Going Up + Right
        # action 7 = [0, 1, 1, 0] Going Down + Left
        # action 8 = [0, 1, 0, 1] Going Down + Right
        # where up, down, left, and right are the AI's actions to move. Each variable will be a binary value of 0 or 1/False or True. 
        return ai_output, h0, c0


class AIHivemindManager: # HIVEMIND TIME!!!
    
    # This will be used later.
    h0 = None
    c0 = None

    # Saving data to be analyzed later.
    total_loss = 0
    total_rewards = 0
    q_value_average = 0
    q_value_max = 0
    q_value_min = 0

    # Amount of data collected (To calculate Average)
    loss_count = 0
    reward_count = 0
    q_value_count = 0

    # Other Important Variables
    ai_save_data = None
    idx = None
    epsilon = init.INITIAL_EPSILON
    loss_fn = nn.SmoothL1Loss()
    resync_counter = 0
    previous_memory = None

    # Uninitilized Objects
    glues = None
    player = None
    ai_list = None

    def __init__(
            self, 
            num_ais, 
            data_manager, 
            model_number, 
            num_layers, 
            hidden_size, 
            output_size, 
            learning_rate, 
            curriculum, 
            window, 
            num_saved_frames,
            sequence_length
        ):
        # Set up the policy and target model.

        self.sequence_length = sequence_length
        self.curriculum = curriculum
        self.num_layers = num_layers
        self.num_ais = num_ais
        self.hidden_size = hidden_size
        self.window = window
        self.memory = torch.zeros((num_saved_frames, sequence_length), dtype=torch.float32).to(init.device)
        self.num_saved_frames=num_saved_frames

        self.policy_model = HivemindLSTM(self.num_layers, sequence_length, self.hidden_size, output_size, num_ais).to(init.device)
        self.model_number = model_number
        self.load_model(init.INFO_FILE, init.SAVE_FOLDER)
        self.target_model = HivemindLSTM(self.num_layers, sequence_length, self.hidden_size, output_size, num_ais).to(init.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)

        self.h0, self.c0 = None, None

        # Save data manager
        self.data_manager = data_manager

    def set(self, glues, player, dim, color):
        self.glues = glues
        self.player = player
        self.ai_list = self.create_ais(self.num_ais, dim, color)
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(init.device)
        self.c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(init.device)

    def create_ais(self, num_ais, dim, color):
        ais = []
        for i in range(num_ais):

            ai_object = AI(
                dim, 
                dim, 
                self.window, 
                color,
                (dim+init.GLUE_DIM)/2,
                self.curriculum,
                objects=[self.player] + self.glues
                )
            ai_object.itr = i
            ais.append(ai_object)
        return ais
    
    def add_frame_to_memory(self, frame):
        frame = frame.to(init.device)

        for i in range(self.num_saved_frames):
            if ((self.memory[self.num_saved_frames - (i+1)] == torch.zeros(self.sequence_length, dtype=torch.float32).to(init.device)).sum().item() == self.sequence_length):
                self.memory[self.num_saved_frames - (i+1)] = frame
                return
        
        self.memory = torch.cat((self.memory[1 : self.num_saved_frames], frame), dim=0)
    
    def update_memory(self):
        x, y = self.player.get_center()
        tensor = torch.tensor([
            x/init.WINDOW_X, 
            y/init.WINDOW_Y,
            self.player.dx/init.MAX_VELOCITY,
            self.player.dy/init.MAX_VELOCITY
            ], dtype=torch.float32)
        
        list_ai_info = []
        for ai in self.ai_list:
            x, y = ai.get_center()
            list_ai_info.append(x/init.WINDOW_X)
            list_ai_info.append(y/init.WINDOW_Y)
            list_ai_info.append(ai.dx/init.MAX_VELOCITY)
            list_ai_info.append(ai.dy/init.MAX_VELOCITY)
            list_ai_info.extend(ai.get_action_vector())
        ai_tensor = torch.tensor(list_ai_info, dtype=torch.float32)
        
        list_glues_location = []
        for glue in self.glues:
            x_glue, y_glue = glue.get_center()
            list_glues_location.append(x_glue/init.WINDOW_X)
            list_glues_location.append(y_glue/init.WINDOW_Y)


        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32)
        tensor = torch.cat((tensor, ai_tensor ,glue_tensor), dim=0)
        tensor = tensor.unsqueeze(0)
        self.add_frame_to_memory(tensor)

    def sync_models(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
    
    def move_ais(self, current_time, sync):

        # Sync the Target model with the Policy models after 10 frames. 
        if self.resync_counter >= sync: 
            self.sync_models()
            self.resync_counter = 0
        q_values, self.h0, self.c0 = self.policy_model(
            self.memory.unsqueeze(0),
            self.h0.detach(),
            self.c0.detach()
        )
        q_values = q_values.squeeze(0)
        self.previous_memory = self.memory

        for ai_index, ai in enumerate(self.ai_list):
            individual_q_value = q_values[ai_index]
            ai.ai_move(individual_q_value, self.epsilon, current_time, self.player)
            ai.check_for_ai_collisions(self.ai_list, pygame.time.get_ticks())

            self.q_value_count += 1
            self.q_value_average += individual_q_value.mean(dtype=torch.float32).item()
            self.q_value_max += individual_q_value.max().item()
            self.q_value_min += individual_q_value.min().item()

    def calculate_reward(self):

        rewards = torch.zeros(self.num_ais)
        for i, ai in enumerate(self.ai_list):
            reward = 0

            # Pentalties AI for being in a glue.
            if ai.in_glue:
                reward += init.GLUE_CONTACT

            # Rewards AI for touching the player.
            if ai.touching_player:
                reward += init.PLAYER_CONTACT

            # Pentalties AI for moving into a wall (Like a bozo).
            if ai.moving_into_wall(axis='x') or ai.moving_into_wall(axis='y'):
                reward += init.WALL_CONTACT

            # Rewards AI for moving towards the player while not in a glue.
            if ai.moving_towards_player(self.player, axis='x') and not ai.in_glue:
                reward += init.MOVING_TOWARDS_PLAYER
            if ai.moving_towards_player(self.player, axis='y') and not ai.in_glue:
                reward += init.MOVING_TOWARDS_PLAYER

            # Checks for collisions between different AI objects and applies a negative reward if there is a collision.
            if ai.collided_with_ai:
                reward += init.AI_COLLISION
                ai.collided_with_ai = False

            # Pentalties AI for not moving.
            if round(ai.dx) == 0 and round(ai.dy) == 0:
                reward += init.NO_MOVEMENT

            ai_x, ai_y = ai.get_center()
            player_x, player_y = self.player.get_center()
            x_diff = ai_x - player_x
            y_diff = ai_y - player_y
            distance = math.sqrt(x_diff**2 + y_diff**2)  # PYTHAGOREAN THEOREM?!?!?!?! YIPPEE!!!!!!!!!!!!!!
            max_distance = ((init.WINDOW_X)**2 + (init.WINDOW_Y) **2)  # PYTHAGOREAN THEOREM AGAIN?!?!?!?!? AMAZING!!!!!!!!
            normalized_distance = distance/max_distance
            reward += math.sqrt(1-(2*normalized_distance))*init.DISTANCE_REWARD_MULTIPLIER


            # Rewards AI for closing the distance between itself and the player.
            # Pentalties AI for increasing the distance between itself and the player.
            if not ai.previous_x == None and not self.player.previous_x == None:

                # These values have a max of 2 and a min of -2 before being transformed to a range of 1 to -1
                x_difference = (abs(ai.previous_x - self.player.previous_x) - abs(x_diff))/init.MAX_VELOCITY
                y_difference = (abs(ai.previous_y - self.player.previous_y) - abs(y_diff))/init.MAX_VELOCITY

                # Now it will be transformed to a range of 1 to -1 and then averaged and added to the reward.
                reward += (x_difference * init.VELOCITY_REWARD_MULTIPLIER  + y_difference * init.VELOCITY_REWARD_MULTIPLIER)/2
                
            ai.in_glue = False
            rewards[i] = reward

        # Increases reward if multiple AIs are touching the player
        num_ais_touching_player = sum([1 for ai in self.ai_list if ai.touching_player])
        if num_ais_touching_player > 1 or self.num_ais == 1:
            rewards += (num_ais_touching_player / self.num_ais)*.75
        
        rewards = torch.clamp(rewards, min=init.MIN_REWARDS, max=init.MAX_REWARDS)

        self.total_rewards += (sum(rewards)/len(rewards)).item()
        self.reward_count += 1

        return rewards

    def save_data(self):
        # Create Rewards
        reward = self.calculate_reward()

        # Select actions
        actions = [ai.action for ai in self.ai_list]

        self.data_manager.append((
            self.previous_memory, 
            actions, 
            self.memory, 
            reward
            ))

    def sequence_training(self, sequences, timesteps_till_backdrop=5):
        
        if sequences is None:
            return 0

        # Get batch and sequence size
        batch_size = len(sequences)
        sequence_length = len(sequences[0])

        # Initilize h0 and c0 cells.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(init.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(init.device)

        # Initilize other variables
        total_loss = 0
        steps_since_backdrop = 0
        accumulated_loss = 0

        for s in range(sequence_length):

            # Get actions and rewards for part of sequence
            actions = torch.tensor([sequences[b][s][1] for b in range(batch_size)], dtype=torch.long).to(init.device)
            rewards = torch.stack([sequences[b][s][3] for b in range(batch_size)]).to(init.device)

            # Get the states and new states.
            states = torch.stack([sequences[b][s][0]for b in range(batch_size)], dim=0)
            new_states = torch.stack([sequences[b][s][2] for b in range(batch_size)], dim=0)

            # Get Q Values and update h0 and c0 cells
            q_values, h0, c0 = self.policy_model(states, h0, c0)

            with torch.no_grad():
                # Get max q values from policy model
                policy_next_q, _, _ = self.policy_model(new_states)
                policy_next_q_max = policy_next_q.argmax(dim=2)

                # Get Target evalutation
                target_next_q, _, _ = self.target_model(new_states)

                next_q_values = torch.zeros_like(rewards)
                for ai_idx in range(self.num_ais):
                    for batch_idx in range(batch_size):
                        action_idx = policy_next_q_max[batch_idx, ai_idx]
                        next_q_values[batch_idx, ai_idx] = target_next_q[batch_idx, ai_idx, action_idx]

                # Calculate the Target using the split max q values. This will be used for training.
                targets = rewards + init.DISCOUNT_FACTOR * next_q_values        

            seq_loss = 0
            for ai_index in range(self.num_ais):
                # Get individual Q_values for each AI object
                q_values_for_ai = q_values[:, ai_index, :]
                actions_for_ai = actions[:, ai_index]
                target_for_ai = targets[:, ai_index]

                # Match Q_values to the chosen action.
                q_values_to_action = torch.stack([q_values_for_ai[b][actions_for_ai[b]] for b in range(batch_size)])

                # Calculate loss using targets and individual AI q_values
                seq_loss += self.loss_fn(q_values_to_action, target_for_ai)

            # Average the total_loss.
            seq_loss /= self.num_ais
            accumulated_loss += seq_loss
            steps_since_backdrop += 1

            if (timesteps_till_backdrop <= steps_since_backdrop) or (s == sequence_length - 1):
                avg_loss = accumulated_loss / steps_since_backdrop

                # Optimize the model
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += avg_loss

                # Detach h0 and c0 cells
                h0 = h0.detach()
                c0 = c0.detach()

                # Reset accumulated loss and steps
                accumulated_loss = 0
                steps_since_backdrop = 0

        avg_loss = total_loss / (sequence_length//timesteps_till_backdrop)
        self.total_loss += avg_loss.item()
        self.loss_count += 1
        return avg_loss

    def load_model(self, file, save_folder):
        # This function checks if one of the saved models can be loaded, and if it can the function will load the model.
        # If not the function will create a new model and save its input shape so it can be saved and used in the future.

        # Check if model exists
        # if not os.path.isdir(file):
        #     print(f"File {file} does not exist!")
        #     raise FileNotFoundError

        with open(file, 'r') as f:
            self.ai_save_data = json.load(f)
        
        empty, idx = self.match_model()
        if idx is None:
            print("Model not found. Creating a new one!")
            if empty:
                self.ai_save_data["Model"].append(1)
                self.model_number = 1
                idx = 0
            elif self.model_number is None:
                self.model_number = get_lowest(self.ai_save_data["Model"])
                idx = len(self.ai_save_data["Model"])
            else:
                idx = len(self.ai_save_data["Model"])
            self.ai_save_data["Model"].append(self.model_number)
            self.ai_save_data["Ais"].append(self.num_ais)
            self.ai_save_data["Glues"].append(int((self.sequence_length - (self.num_ais * 13 + 4))/2))
            self.ai_save_data["Hidden"].append(self.hidden_size)
            self.ai_save_data["Frames"].append(self.num_saved_frames)
            self.ai_save_data["Layers"].append(self.num_layers)
            self.ai_save_data["Epsilon"].append(init.INITIAL_EPSILON)
            self.idx = idx
            return
        self.idx = idx
        
        print(f"Found model {self.model_number}, Now loading model...")
        model_dir = save_folder + '/model_' + str(self.model_number) + ".pth"
        save_dict = torch.load(model_dir)
        self.policy_model.load_state_dict(save_dict)
        print(f"Model {self.model_number} successfully loaded!")
        
    def save_model(self, file, save_folder):
        # Saves the model 

        with open(file, 'w') as f:
            json.dump(self.ai_save_data, f)

        print("Saving model")
        torch.save(self.policy_model.state_dict(), save_folder + "/" + "model_" + str(self.model_number) + ".pth")
        print(f"Model {self.model_number} saved successfully.\n")


    def match_model(self):

        if len(self.ai_save_data["Model"]) == 0:
            return True, None
            
        for i in range(len(self.ai_save_data["Model"])):
            ais = self.ai_save_data["Ais"][i] == self.num_ais
            glues = self.ai_save_data["Glues"][i] == init.GLUES
            hidden = self.ai_save_data["Hidden"][i] == self.hidden_size
            frames = self.ai_save_data["Frames"][i] == self.num_saved_frames
            layers = self.ai_save_data["Layers"][i] == self.num_layers

            if ais and glues and hidden and frames and layers:
                self.model_number = self.ai_save_data["Model"][i]
                self.epsilon = self.ai_save_data["Epsilon"][i]
                idx = i
                return False, idx

        return False ,None