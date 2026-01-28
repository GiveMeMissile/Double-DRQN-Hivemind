#######################################################################################################
# File Name:
#   DataManager
#
# Purpose:
#   To manage the training data and progress data during training
#   and data generation. Saves training and progress data for later
#   use and loads training data for training.
#
# Used:
#   1 - Used by Training to load training data.
#   2 - Used by main to save progress data and training data.
#   3 - Used by DataExtractor to analyze progress data and measure preformance.
#######################################################################################################

import os
import random
import json
import math
import utils
import pickle
import torch
import pandas as pd
import numpy as np
import Config as init
import matplotlib.pyplot as plt
from collections import deque


class SequentialTrainingData:
    # This class will store some of the training data for future training WOHOOOOOOO!!!
    def __init__(self, save_folder, info_file, model_number=None, max_episodes=init.MAX_EPISODES):
        if max_episodes < 50:
            max_episodes = 50
        self.episodes = deque([], maxlen=max_episodes)
        self.max_episodes = max_episodes
        self.save_folder = save_folder
        self.info_file = info_file
        self.current_episode = []
        self.model_number = model_number
        self.preloaded_episodes = 0

    def append(self, transition):
        # A transition is a tuple which contains information used for training the AI.
        # Transition = (AI memory, action made by AI, New memory after action was made, Reward earned from the action)
        
        self.current_episode.append(transition)

    def episode_end(self):
        # Saves the current episode to the episode deque and resest the current episode.

        if len(self.current_episode) > 0:
            self.episodes.append([self.current_episode, True])
            self.current_episode = []

    def get_max_episode_length(self):
        # Gets the length of the longest saved episode.

        max = 0
        for eps in self.episodes:
            if (len(eps[0]) > max):
                max = len(eps[0])
        return max

    def get_sample_sequence(self, batch_size, sequence_length=20):
        # Produces a sequence to be used for training.

        if len(self.episodes) == 0:
            return None
        
        data = self.get_data()
        
        sequences = []
        if self.get_max_episode_length() < sequence_length:
            sequence_length = self.get_max_episode_length()

        for _ in range(batch_size):
            episode = random.choice(data)

            # Ensure that episode >= sequence length
            count = 0
            while (len(episode) < sequence_length):
                episode = random.choice(data)
                count += 1
                if count > 200:
                    break
            if count > 200:
                continue
            
            # Get sequence from episode
            start_idx = random.randint(0, len(episode) - sequence_length - 1)
            # print(f"Eps Len: {len(episode)} | Start: {start_idx} | End: {start_idx + sequence_length}")
            sequence = self.prepare_sequence(episode, start_idx, start_idx + sequence_length)
            # print(f"Sequence Length: {len(sequence)}")
            sequences.append(sequence)

        # Return sequences only if it equals batch size
        # print(f"Sequence Batch Length: {len(sequences)}")
        if len(sequences) == batch_size:
            return sequences
        return None
    
    def prepare_sequence(self, episode, start_idx, end_idx):
        # Prepares a sequence to be ready for training.

        prepared_sequence = []
        for i in range(start_idx, end_idx):
            prev_memory = self.reassemble_memory(episode, i, 0, init.NUM_SAVED_FRAMES, init.SEQUENCE_LENGTH)
            actions = episode[i][1]
            memory = self.reassemble_memory(episode, i, 2, init.NUM_SAVED_FRAMES, init.SEQUENCE_LENGTH)
            rewards = episode[i][3]
            prepared_sequence.append((prev_memory, actions, memory, rewards))

        return prepared_sequence

    def reassemble_memory(self, data, memory_idx, part_idx, memory_len, tensor_len):
        # Reassebles the tensor to fit what the AI expects as its input.

        memory_list = []
        min = memory_idx-memory_len+1

        for i in range(min, memory_idx+1):
            if i >= 0:
                # try:
                memory_list.append(data[i][part_idx])
                # except IndexError:
                #     memory_list.append(data[len(data)-1][part_idx])
            else:
                memory_list.append(torch.zeros(tensor_len).to(init.device))

        memory = torch.stack(memory_list, dim=0)
        return memory
    
    def batch_data(self, batch_size):
        # Gets a batch of data ig (I have no idea what this function does...)

        data = list(self.episodes)
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
        return batches
    
    def save_data(self, episodes, create=False):
        # Saves the currently loaded data

        print("Saving training data...")
        self.to_device("cpu")
        idx = self.check_for_similars()
        saved = False
        if idx == -1 or create:
            self.create_new_file(episodes)
            return None
        saved = self.append_to_file(idx, episodes)
        if not saved:
            print("Data cannot be appended, will create a new data file...")
            self.create_new_file(episodes)
        print("Data Saved!!!")

    def append_to_file(self, idx, episodes):
        # Appends currently loaded data to another file

        # print(self.episodes[0][0][0][0])
        metadata = self.get_metadata()
        filename = self.get_file(idx)
        if filename is None:
            return False
        with open(filename, "rb") as f:
            old_data = pickle.load(f)

        new_data = []
        for episode in self.episodes:
            if episode[1]:
                new_data.append(episode[0])

        data = old_data + new_data
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        metadata["Episodes"][idx] += episodes
        with open(self.info_file, "w") as f:
            json.dump(metadata, f)

        return True

    def create_new_file(self, episodes, number=None):
        # Creates a new pickle file and saved all of the current data onto that file.

        metadata = self.get_metadata()
        if number is None:
            number = utils.get_lowest(metadata["Data Number"])
        filename = self.save_folder + "/" + "training_data_" + str(number) + ".pkl"
        self.add_metadata(number, episodes)
        open(filename, 'x')
        with open(filename, 'wb') as f:
            pickle.dump(self.get_data(), f)

    def get_data(self):
        data = []
        for episode in self.episodes:
            data.append(episode[0])
        return data

    def add_metadata(self, number, episodes):
        # Adds metadata to json file when creating a new Training Data File.

        metadata = self.get_metadata()
        metadata["Data Number"].append(number)
        metadata["Ais"].append(init.NUM_AI_OBJECTS)
        metadata["Glues"].append(init.GLUES)
        metadata["Saved Frames"].append(init.NUM_SAVED_FRAMES)
        metadata["Window X"].append(init.WINDOW_X)
        metadata["Window Y"].append(init.WINDOW_Y)
        metadata["Episodes"].append(episodes)
        with open(self.info_file, "w") as f:
            json.dump(metadata, f)
    
    def check_for_similars(self):
        # Checks if any files are similar to the current enviorment data
        
        metadata = self.get_metadata()
        for i in range(len(metadata["Data Number"])):
            ais = True if metadata["Ais"][i] == init.NUM_AI_OBJECTS else False
            glues = True if metadata["Glues"][i] == init.GLUES else False
            saved = True if metadata["Saved Frames"][i] == init.NUM_SAVED_FRAMES else False
            win_x = True if metadata["Window X"][i] == init.WINDOW_X else False
            win_y = True if metadata["Window Y"][i] == init.WINDOW_Y else False
            if ais and glues and saved and win_x and win_y:
                return i
        return -1

    def get_file(self, idx):
        # Takes in a file index and returns the file name

        metadata = self.get_metadata()
        try:
            number = metadata["Data Number"][idx]
        except IndexError:
            print("\nThe index does not exist, no file matches the index\n")
            return None
        name = self.save_folder + "/" + "training_data_" + str(number) + ".pkl"
        return name

    def get_metadata(self):
        # Gets and returns metadata for all saved training data.

        with open(self.info_file, "r") as f:
            return json.load(f)
        
    def to_device(self, device):
        # Transfers all loaded data to the set device.

        if not (device == "cpu" or device == "cuda"):
            return False
        
        for i in range(len(self.episodes)):
            for j in range(len(self.episodes[i][0])):
                self.episodes[i][0][j][0] = self.episodes[i][0][j][0].to(device)
                self.episodes[i][0][j][2] = self.episodes[i][0][j][2].to(device)

    def load_data(self, num_episodes=None, number=None):
        # Loads the data from a certian index of the metadata and saved it to the current episodes.

        idx = 0

        if number is None:
            idx = self.check_for_similars()
        else:
            idx = utils.get_idx_from_number(self.get_metadata(), number, "Data Number")

        if idx == -1:
            print(f"Error: Data Number {number} does not exist")
            return

        filename = self.get_file(idx)
        data = self.get_data_from_file(filename)
        random.shuffle(data)
        if num_episodes is None:
            num_episodes = len(data) if len(data) <= self.max_episodes else self.max_episodes
        print(f"Num Episodes: {num_episodes} | Len: {len(data)}")
        for i in range(num_episodes):
            self.episodes.append([data[i], False])
        self.to_device(init.device)

    def get_data_from_file(self, filename):
        # Returns data from a pickle file.

        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def __len__(self):
        return len(self.episodes)
    

class ProgressTracker:
    # Keeps track of important data to be displayed and saved them at the end.

    data = {
        "Episodes": [],
        "Rewards": [],
        "Actions": [],
        "Action Spread": [],
        "Health": [],
        "Loss": [],
        "Epsilon": [],
        "Time": [],
        "Q Values Max": [],
        "Q Values Average": [],
        "Q Values Min": [],
        "Q Values Train": []
    }

    model_number = None

    def __init__(self, iteration):
        self.iteration = iteration

    def append(self, item, location):
        try:
            self.data[location].append(item)
        except Exception:
            print("Invalid location, please input a valid location.")
    
    def __len__(self):
        return len(self.data["Episodes"])
    
    def calculate_sd(self, data, mean):
        # Calculate the Mean Absolute Diviation of the Data: sd^2 = (∑((value - mean)^2))/total

        sd = 0

        for value in data:
            sd += (value - mean)**2
        
        sd /= len(data)
        sd = math.sqrt(sd)
        return sd

    def calculate_mean(self, data):
        return sum(data)/len(data)

    def calculate_z_scores(self, data, mean, sd):
        # Calculate the z_scores of the data, equation: z_score = (value - mean_of_data)/standard deviation

        z_scores = []
        for value in data:
            try:
                z_score = (value-mean)/sd
            except ZeroDivisionError:
                z_score = 0
            z_scores.append(z_score)

        return z_scores
    
    def save_as_cvs(self):
        # Convert any potential tensors to Python numbers
        cleaned_data = {
            "Episodes": [float(x) if hasattr(x, 'item') else x for x in self.data["Episodes"]],
            "Rewards": [float(x) if hasattr(x, 'item') else x for x in self.data["Rewards"]],
            "Actions": [float(x) if hasattr(x, 'item') else x for x in self.data["Actions"]],
            "Action Spread": [float(x) if hasattr(x, 'item') else x for x in self.data["Action Spread"]],
            "Health": [float(x) if hasattr(x, 'item') else x for x in self.data["Health"]],
            "Loss": [float(x) if hasattr(x, 'item') else x for x in self.data["Loss"]],
            "Epsilon": [float(x) if hasattr(x, 'item') else x for x in self.data["Epsilon"]],
            "Time": [float(x) if hasattr(x, 'item') else x for x in self.data["Time"]],
            "Q Values Max": [float(x) if hasattr(x, 'item') else x for x in self.data["Q Values Max"]],
            "Q Values Average": [float(x) if hasattr(x, 'item') else x for x in self.data["Q Values Average"]],
            "Q Values Min": [float(x) if hasattr(x, 'item') else x for x in self.data["Q Values Min"]],
            "Q Values Train": [float(x) if hasattr(x, 'item') else x for x in self.data["Q Values Train"]]
        }
        
        file_name = init.DATA_FOLDER + "/" + "model_" + str(self.model_number) + "_" + str(self.iteration) +"_data.csv"
        df = pd.DataFrame(cleaned_data)

        # If no csv file exists for this save, then create a new one and return
        if not os.path.exists(file_name):
            df.to_csv(file_name, index=False)
            return
        
        # If there is already a csv file with data in it, then we merge the data
        old_df = pd.read_csv(file_name)
        # Fix the concatenation
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(file_name, index=False)

    def graph(self, data, names):
        # Takes in a list of data and graphs all of the data vs the episodes

        for i in range(len(data)):
            values = np.array(data[i])
            plt.plot(self.episodes, values, label = names[i])
        
        plt.legend()
        plt.show()
            

    def get_info(self, data):
        mean = self.calculate_mean(data)
        sd = self.calculate_sd(data, mean)
        z_scores = self.calculate_z_scores(data, mean, sd)
        return mean, sd, z_scores

    def graph_results(self):
        self.episodes = self.data["Episodes"]
        _, _, r_z_scores = self.get_info(self.data["Rewards"])
        _, _, h_z_scores = self.get_info(self.data["Health"])
        _, _, e_z_scores = self.get_info(self.data["Epsilon"])
        _, _, l_z_scores = self.get_info(self.data["Loss"])
        _, _, t_z_scores = self.get_info(self.data["Time"])
        z_score_list = [r_z_scores, h_z_scores, e_z_scores, l_z_scores, t_z_scores]
        name_list = ["Rewards", "Health", "Epsilon", "Loss", "Time"]
        self.graph(z_score_list, name_list)


class DataAnalyzer:
    # A class which is used to graph and collect data from csv files.

    data = []

    def __init__(self, folders=None):
        if not folders is None:
            self.folder = folders

    def get_csv_data(self, filename, append=True, itr=0):
        # This reads a csv file and appends the data to the dictionary.

        file_exists = False
        file_folder = None

        for folder in self.folder:
            if not os.path.isfile(folder + "/" + filename):
                print("File does not exist.")
            else:
                file_exists = True
                file_folder = folder
                break
        
        if not file_exists:
            print("File does not exist in any of the given folders.")
            return None
        
        df = pd.read_csv(file_folder + "/" + filename)
        csv_data = df.to_dict(orient="list")

        if append or self.data == []:
            self.data.append(csv_data)
            return csv_data

        for key in self.data.keys():
            if key in csv_data[itr]:
                self.data[itr][key] = csv_data[key]
            else:
                print(f"Key {key} not found in CSV file.")

        return self.data

    def append(self, item, location):
        try:
            self.data[location].append(item)
        except Exception:
            print("Invalid location, please input a valid location.")
    
    def __len__(self):
        return len(self.data["Episodes"])
    
    def calculate_sd(self, data, mean):
        # Calculate the Standard Deviation of the Data: sd^2 = (∑((value - mean)^2))/total

        sd = 0

        for value in data:
            sd += (value - mean)**2
        
        sd /= len(data)
        sd = math.sqrt(sd)
        return sd

    def calculate_mean(self, data):
        return sum(data)/len(data)
    
    def calculate_range(self, data):
        return max(data) - min(data)
    
    def calculate_mad(self, data, mean):
        # Calculate the Mean Absolute Diviation of the Data: mad = (∑|value - mean|)/total

        mad = 0

        for value in data:
            mad += abs(value - mean)
        
        mad /= len(data)
        return mad
    
    def calculate_median(self, data):
        # Calculate the median of the data

        sorted_data = sorted(data)
        length = len(data)

        if length % 2 == 0:
            median = (sorted_data[length//2 - 1] + sorted_data[length//2]) / 2
        else:
            median = sorted_data[length//2]
        
        return median

    def calculate_z_scores(self, data, mean, sd):
        # Calculate the z_scores of the data, equation: z_score = (value - mean_of_data)/standard deviation

        z_scores = []
        for value in data:
            try:
                z_score = (value-mean)/sd
            except ZeroDivisionError:
                z_score = 0
            z_scores.append(z_score)

        return z_scores
    
    def calculate_lsrl(self, x, y):
        # Calculates the lsrl of the data, which is the linear best fit line for the data.
        # The equation is ŷ = mx + b, where ŷ is the predicted y value, m is the slope, and b is the y-int.
        # m = r(sdy/sdx) where r = (1/total)∑(((x_mean - x)/sdx)*(y_mean - y)/sdy))
        # b = y_mean - m*x_mean

        y_mean = self.calculate_mean(y)
        x_mean = self.calculate_mean(x)
        sdy = self.calculate_sd(y, y_mean)
        sdx = self.calculate_sd(x, x_mean)
        total_sum = sum([((yi - y_mean)/sdy)*((xi - x_mean)/sdx) for xi, yi in zip(x, y)])
        r = (1/len(x)) * total_sum
        m = r * (sdy/sdx)
        b = y_mean - m*x_mean
        return m, b, r
    
    def graph_scatter(self, x, y, x_name, y_name):
        # Takes in a list of data from two different variables and graphs all of the data vs the episodes

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label = f"{y_name} vs {x_name}")
        
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(f"{x_name} vs {y_name} Scatter Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graph_lsrl(self, x, y, x_name, y_name, scatter=False):
        # Takes in two lists, x and y, and calculates and graphs the lsrl for the graph.

        if len(x) != len(y):
            print(f"List size Error: y_list: {len(x)}, x_list = {len(y)}. Please input correct list sizes")
            return

        m, b, r = self.calculate_lsrl(x, y)
        pred_y = [m*xi + b for xi in x]
        if scatter:
            plt.scatter(x, y, color='blue', label="Original Values")
        plt.plot(x, pred_y, color='red', label=f"LSRL: y = {m:.2f}x + {b:.2f}, r = {r:.2f}")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(f"{x_name} vs {y_name} with LSRL")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graph(self, x, data, names, x_name):
        # Takes in a list of data from different variables and graphs all of the data vs the episodes

        if type(data) != list:
            data = np.array([data])
        if type(names) != list:
            names = [names]

        plt.figure(figsize=(10, 6))
        for name, values in zip(names, data):
            values = np.array(values)
            plt.plot(x, values, label = name)
        
        plt.xlabel(x_name)
        plt.ylabel("Value")
        plt.title(f"{x_name} vs {", ".join([label for label in names])} Graph")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def graph_box_plot(self, data, name):
        # Takes in a list of data from a single variable and graphs all of the data vs the episodes

        fig, ax = plt.subplots()
        data = np.array(data).reshape(-1, 1)
        bp = ax.boxplot(data)

        ax.set_xticklabels([name])
        plt.ylabel("Value")
        plt.title(f"{name} Box Plot")

        plt.show()

    def graph_histogram(self, data, name, bins=10):
        # Takes in a list of data from a single variable and graphs all of the data vs the episodes

        data = np.array(data)
        plt.hist(data, bins=bins, label=[name])
        plt.legend()
        plt.title(f"{name} Histogram")
        plt.show()

    def get_info(self, data):
        mean = self.calculate_mean(data)
        sd = self.calculate_sd(data, mean)
        z_scores = self.calculate_z_scores(data, mean, sd)
        return mean, sd, z_scores
    
    def get_linear_vals(self, length):
        return [i + 1 for i in range(length)]