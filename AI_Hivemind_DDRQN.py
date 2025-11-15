# Note: I am SO CLOSE (I think lol)

# Todo: 

# 1: Temporal learning: Allow the LSTM to make use of its h0 and c0 cells by adding sequence length.
#   This will allow the AI to have a better understading of the enviorment, actions, and more.
#   It will allow the LSTM to unlock its TRUE POTENTIAL!!!

# 2: Training Data Saving: Save information which will be used to train the AI outside the game program.
#   This will allow episode data to be saved into a csv or pickle file to be used later.
#   This will allow for faster training and quicker testing of hyperparameters. 

# 3: Epsilon Alternating: Alternate between regular and bias epsilon (based on episode). More VARAITY

# 4: Iterations: This is the final step (I hope). I will iterate through the following steps below until the program works as intended. (In Progress)
#   a): Train the AI for a varying amount of episodes (10-400)
#   b): Preform data analysis on the results of the training.
#   c): Change hyperparameters and/or reward values based on the data analysis.
#   d): Repeat until the AI is functioning as intended.

# 5: Get happy... (Impossible)

# Iteration logs:
#   Iter 1: eps: 190, ais: 3, glues: 0, lr: 0.0001, Other Changes: DNE
#   Iter 2: eps: 150, ais: 3, glues: 5, lr: 0.0001, Other Changes: Minor Tweak to Reward values.
#   Iter 3: eps: 150, ais: 1, glues: 5, lr: 0.00001, Other Changes: DNE
#   Iter 4: eps: 150, ais: 1, glues: 5, lr: 0.000001, Other Changes: Hidden Size increase 64x2
#   Iter 5: eps: 150, ais: 1, glues: 5, lr: 0.0000001, Other Changes: Hidden Size increase 64x4
#   Iter 6: eps: 150, ais: 1, glues: 5, lr: 0.000001, Other Changes: Epsilon Update to fix issue #1.
#   Iter 7: eps: 150, ais: 1, glues: 5, lr: 0.000001, Discount: 0.995, Other Changes: Added Curriculum Learning and increased Discount Factor.
#   Iter 8: eps: 150, ais: 1, glues: 5, lr: 0.000001, Discount: 0.97, Other Changes: Added post game learning.
#   Iter 9: eps: 150, ais: 1, glues: 5, lr: 0.000001, Discount: 0.975, Other Changes: Epsilon Bias Addition.
#   Iter 10: eps: 15, ais: 1, glues: 5, lr: 0.000001, Discount: 0.975, Other Changes: Hidden Size increase 64*8
#   Iter 11: eps: 15, ais: 1, glues: 5, lr: 0.000001, Discount: 0.975, Other Changes: Increased Wall punishment.
#   Iter 12: eps: 150, ais: 3, glues: 0, lr: 0.00001, Discount: 0.975, Other Changes: Decreased initial epsilon and added display info.
#   Iter 13: eps: 150, ais: 1, glues: 0, lr: 0.0000001, Discount: 0.975, Other Chages: Added a distance based Rewards and added reward capping.
#   Iter 14: eps: 150, ais: 1, glues: 0, lr: 0.0000001, Discount: 0.975, Other Changes: Smaller screen, no motion of player, and random spawn location of player.
#   Iter 15: eps: 150, ais: 1, glues: 0, lr: 0.0000001, Discount: 0.975, Other Changes: Same as 14, but removed change and distance rewards
#   Iter 16: eps: 150, ais: 1, glues: 0, lr: 0.001, Discount: 0.975, Other Changes: Revert iter 15 changes.
#   Iter 17: eps: 150, ais: 1, glues: 0, lr: 0.001, Discount: 0.975, Other Changes: Removed training with c0 and h0 cells and removed detatchment of said cells.
#   Iter 18: eps: 150, ais: 1, glues: 0, ir: 0.001, Discount: 0.975, Other Changes: Added q_value Tracking for debugging.
#   Iter 18.5: Same as iter 18, but also changed tracking of actions, and added minimum q_values tracking.
#   Iter 19: Iter 18 values, but added actions into the LSTM input, and disabled biased epsilon.
#   Iter 20: I am so dumb, lr = 0.001 (for reals this time)
#   Iter 21: Bias epsilon turned back on.
#   Iter 21.5: eps: 10, discount = 0.92, model_resync = 200, add target capping [-100, 100]
#   Iter 22: eps: 10, discount = 0.92, model_resync = 200, add Double DQN to training.
#   Iter 23: eps: 150, ais: 1, glues: 5, lr: 0.001, Discount: 0.92, Other changes: Revert to old setup (Testing old setup for fun).
#   Iter 24: eps: 10, ais: 1, glues: 0, lr: 0.0001, Discount: 0.92, Other changes: Revert to iter 22 game settings.
#   Iter 25: eps: 10, ais: 1, glues: 0, lr: 0.001, Discount: 0.9, Other changes: Removed Clamping of Targets.
#   Iter 26: eps: 10, ais: 1, glues: 0, lr: 0.0001, Discount: 0.9, Other changes: None
#   Iter 26.5: Same as 26, but added Target Clampings back [-50, 50]
#   Iter 27: eps: 150, ais: 1, glues: 0, lr: 0.0001, Discount: 0.9, Other changes: Bosted action based rewards, removed clamping, thats it ig.
#   Iter 28: eps: 150, ais: 1, glues: 0, lr: 0.0001, Discount: 0.9, Other Changes: Improved Model architecture, inceasing the size of the split layers.
#   Iter 29: eps: 150, ais: 1, glues: 5, lr: 0.0001, Discount: 0.9, Other Changes: Reversion to original training structure, halfed hidden size.
#   Iter 30: eps: 150, ais: 2, glues: 5, lr: 0.0001, Discount: 0.9, Other Changes: Nothing, checking AI Hivemind preformance!!!


import torch
import os
import pygame
import random
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from collections import deque

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
GLUES = 1
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

# AI Constants
NUM_LAYERS = 4
COLLISION_TIMER = 500
HIDDEN_SIZE = 64*4
OUTPUT_SIZE = 9
LEARNING_RATE = 0.0001
NUM_AI_OBJECTS = 1
NUM_SAVED_FRAMES = 20
SEQUENCE_LENGTH = 4 + (13 * NUM_AI_OBJECTS) + (2 * GLUES) # 4 for player, 13 for each AI, 2 for each glue.
INPUT_SHAPE = (NUM_SAVED_FRAMES, SEQUENCE_LENGTH)
DISCOUNT_FACTOR = 0.90
SYNC_MODEL = 200
GAME_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS = 10
TRAINING_SYNC = 10
EPSILON_ACTIVE = True # Determines if epsilon is active
BIASED_EPSILON = True
EPSILON_TIME_MAX, EPSILON_TIME_MIN = 500, 100
TRAINING_INCREMENT = 100 # How many episodes/games the ai plays until values return to the original
EPSILON_DECAY = 5
INITIAL_EPSILON = 1
AI_SAVE_DATA = {
    "Model": [],
    "Ais": [],
    "Glues": [],
    "Hidden": [],
    "Frames": [],
    "Layers": [],
    "Epsilon": []
}

# File Constants
SAVE_FOLDER = "RL_LSTM_Models"
INFO_FILE = SAVE_FOLDER + "/" +"model_info.json"
DATA_FOLDER = "RL_LSTM_Progress_Data"

# AI Reward values, reward values should in the interval [-10, 10]. If rewards exceed the interval, they will just be the max/min of the interval.
MIN_REWARDS = -10
MAX_REWARDS = 10
VELOCITY_REWARD_MULTIPLIER = 3
DISTANCE_REWARD_MULTIPLIER = 2
PLAYER_CONTACT = 10
GLUE_CONTACT = -5
WALL_CONTACT = -3
MOVING_TOWARDS_PLAYER = 3
NO_MOVEMENT = -8
AI_COLLISION = -5 

# Other important stuff
iteration = -1  # Used for data saving and testing purposes.
training = True  # If AI is going to be actively training (if true then activiates curriculum, data saving, and post game training)
delete_model_file = False # If True then if a model file exists for the current variables, it gets deleated and replaced by a new model.
device = "cuda" if torch.cuda.is_available() else "cpu"  # Device agnostic code ig
window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))  # Pygame window wow
previous_time = 0  # Used for time calculation for each game/episode.
pygame.init()  # Why am I commenting on all of these lines of code?
pygame.font.init()  # I am so gooberish ig idk please turn me into a fish.
font = pygame.font.SysFont("New Roman", 30)  # Its font idk what to say
training_font = pygame.font.SysFont("New Roman", 50)  # font, but for training (:


class Object:
    # Base class for all objects in the game. It has a hitbox, velocity, color, and more ig.

    def __init__(self, width, height, window, color, distance, objects=None, x=None, y=None):
        self.width = width
        self.height = height
        self.dx = 0
        self.dy = 0
        self.color = color
        self.window = window
        self.amplifier = 1
        if x is None and y is None:
            x, y = self.find_valid_location(objects, distance)
        elif x is None:
            x, _ = self.find_valid_location(objects, distance)
        elif y is None:
            _, y = self.find_valid_location(objects, distance)
        self.hitbox = pygame.Rect(x, y, width, height)

    def display(self):
        pygame.draw.rect(self.window, self.color, self.hitbox)

    def move(self):
        self.hitbox.x += self.dx
        self.hitbox.y += self.dy

    def location(self):
        return self.hitbox.x, self.hitbox.y
    
    def check_bounds(self):
    # Checks for collisions with the boarders of the screen and it inverts the velocity. No going of the screen, Tehe.

        if self.hitbox.x < 0:
            self.hitbox.x = 0
            self.dx = -self.dx/2
        elif self.hitbox.x > WINDOW_X - self.width:
            self.hitbox.x = WINDOW_X - self.width
            self.dx = -self.dx/2

        if self.hitbox.y < 0:
            self.hitbox.y = 0
            self.dy = -self.dy/2
        elif self.hitbox.y > WINDOW_Y - self.height:
            self.hitbox.y = WINDOW_Y - self.height
            self.dy = -self.dy/2

        if self.hitbox.x < -25:
            self.hitbox.x += 100
        elif self.hitbox.x > WINDOW_X + 25:
            self.hitbox.x -= 100

        if self.hitbox.y < -25:
            self.hitbox.y += 100
        elif self.hitbox.y > WINDOW_Y + 25:
            self.hitbox.y -= 100

    def random_move(self):
        # Selects a random direction for the object to move in.
        x = random.randint(-1, 1)
        y = random.randint(-1, 1)
        return x, y

    def apply_friction(self):

        if self.dx > 0:
            self.dx -= FRICTION
        elif self.dx < 0:
            self.dx += FRICTION
        if self.dy > 0:
            self.dy -= FRICTION
        elif self.dy < 0:
            self.dy += FRICTION
        
        if self.dx < FRICTION and self.dx > -FRICTION:
            self.dx = 0
        if self.dy < FRICTION and self.dy > -FRICTION:
            self.dy = 0
            
    
    def check_max_velocity(self):
        if self.dx > MAX_VELOCITY:
            self.dx = MAX_VELOCITY
        elif self.dx < -MAX_VELOCITY:
            self.dx = -MAX_VELOCITY
        if self.dy > MAX_VELOCITY:
            self.dy = MAX_VELOCITY
        elif self.dy < -MAX_VELOCITY:
            self.dy = -MAX_VELOCITY

    # Math is FUN

    def get_center(self):
        # Returns the center of the object.
        return self.hitbox.x + self.width/2, self.hitbox.y + self.height/2
    
    def find_valid_location(self, objects, distance):
        # This function finds a random location where the object is not on top of another object.  
        count = 0
        x, y = 0, 0
        not_valid = True
        while not_valid:
            x = random.randint(0, WINDOW_X - int(self.width/2))
            y = random.randint(0, WINDOW_Y - int(self.height/2))
            if objects is None:
                break
            if len(objects) == 0:
                break
            for obj in objects:
                obj_x, obj_y = obj.get_center()
                if (x > obj_x - distance) and (x < obj_x + distance):
                    break
                elif (y > obj_y - distance) and (y < obj_y + distance):
                    break
                else:
                    not_valid = False
            count += 1
            if count >= 150:
                break

        return x - self.width/2, y - self.height/2


class Glue(Object):
    # Simple obstacle that is an issue for both the player and AI.

    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        super().__init__(width, height, window, color, distance, objects, x, y)
        self.timer = 0
        self.glue_drag = 0.5
        self.amplifier = 10
        self.movement_timer = 0

    def alter_velocity(self, dx, dy):
        # This function is used to alter the velocity of the object when it collides with the glue.

        if dx > 0:
            dx -= self.glue_drag * dx
        elif dx < 0:
            dx -= self.glue_drag * dx
        
        if dy > 0:
            dy -= self.glue_drag * dy
        elif dy < 0:
            dy -= self.glue_drag * dy
        
        return dx, dy

    def check_for_collisions(self, objects, current_time):
        for obj in objects:
            # Managing collisions with da glue.
            if self.hitbox.colliderect(obj.hitbox):
                
                obj.dx, obj.dy = self.alter_velocity(obj.dx, obj.dy)
                if not (self.dx == 0 and self.dy == 0):
                    obj.dx += (self.glue_drag * self.dx)
                    obj.dy += (self.glue_drag * self.dy)
                
                if isinstance(obj, AI):
                    obj.in_glue = True
                
                if not isinstance(obj, Player):
                    continue
                obj.contacted_object = True
                obj.objects.append(self)
                obj.remove_objects_timer = current_time

                # Damages the player for colliding with glue.
                if current_time - self.timer >= DAMAGE_COOLDOWN:
                    obj.health -= 1
                    self.timer = current_time

                          
class Player(Object):
    x_direction = 0
    y_direction = 0
    contacted_object = False
    objects = []
    remove_objects_timer = 0
    change_direction_timer = 0
    change_direction_time = 0
    previous_x = None
    previous_y = None
    health = 30

    def __init__(self, width, height, window, color, objects, distance, curriculum, x=None, y=None):
        super().__init__(width, height, window, color, objects, distance, x, y)
        self.override = not RANDOM_MOVE
        self.curriculum = curriculum

    def player_move(self, current_time):
        # Player movement using WASD keys. The player can move in all directions and has a maximum velocity.

        keys = pygame.key.get_pressed()
        # Checking for key clicks and adding the proper acceleration to the velocity.

        self.previous_x, self.previous_y = self.get_center()

        if keys[pygame.K_w]:
            self.dy -= ACCELERATION
            self.override = True
        if keys[pygame.K_s]:
            self.dy += ACCELERATION
            self.override = True
        if keys[pygame.K_a]:
            self.dx -= ACCELERATION
            self.override = True
        if keys[pygame.K_d]:
            self.dx += ACCELERATION
            self.override = True
        
        self.apply_friction()
        if not training or self.override:
            self.check_max_velocity()
        else:
            max = round(self.curriculum.player_max)
            if max <= self.dx:
                self.dx = max
            elif -max >= self.dx:
                self.dx = -max
            if max <= self.dy:
                self.dy = max
            elif -max >= self.dx:
                self.dy = -max
            if max > MAX_VELOCITY:
                self.curriculum.player_max = MAX_VELOCITY

        # If the player is controlling the square as known by self.override equaling true. 
        # Then we simply move the object and check the boundries then return ending the function.
        if self.override:
            self.move()
            self.check_bounds()
            return

        # If self.override = False and the player square has not collided with any objects.
        # Then the square moves randomly via the random_move() function
        if current_time - self.change_direction_timer >= self.change_direction_time:
            self.x_direction, self.y_direction = self.random_move()
            self.change_direction_timer = current_time
            self.change_direction_time = random.randint(PLAYER_TIME_MIN, PLAYER_TIME_MAX)
        if not self.contacted_object:
            if self.curriculum.player_movement < random.random():
                self.move()
                self.check_bounds()
                return
            self.dx += ACCELERATION * self.x_direction
            self.dy += ACCELERATION * self.y_direction
            self.move()
            self.check_bounds()
            return
        
        # If there is a collision with 1 or more objects then the player object will accelerate away from the object(s) which collided with it.
        # This allows for better automated movement of the player square when it is not being controlled by the player.
        avg_x, avg_y = self.get_objects_average()
        x, y = self.location()
        if avg_x < x:
            self.dx += ACCELERATION
        else:
            self.dx -= ACCELERATION

        if avg_y < y:
            self.dy += ACCELERATION
        else:
            self.dy -= ACCELERATION

        # This if statement below will make the player square continue to move away from the contacted objects
        # for 1/4 of a second so the player square will make some distance between the object it collided with and itself.
        if current_time - round(self.curriculum.player_reaction_time) >= self.remove_objects_timer:
            self.contacted_object = False
            self.objects.clear()

        # Calling the move() and check_bounds() functions if random_move() is not called.
        self.move()
        self.check_bounds()


    def get_objects_average(self):
        # This function returns the average x and y coords of the center of all collided objects.
        # This will be utilized to decide the direction the player will move when it is collided with multiple objects.

        average_obj_x = 0
        average_obj_y = 0
        for obj in self.objects:
            x, y = obj.location()
            average_obj_x += x
            average_obj_y += y

        return average_obj_x/len(self.objects), average_obj_y/len(self.objects)
    
    # Never underestimate a fish


class AI(Object):
    # This is the object which the neural network will control. It is the AI which will hunt the player.

    itr = None
    collided_with_ai = False
    ai_collision_timer = 0
    action_reset_timer = 0
    action_reset_time = 0
    timer = 0
    change_direction_timer = 0

    in_glue = False
    touching_player = False
    action = 0
    epsilon_action = None
    previous_x = None
    previous_y = None

    action_distribution = []
    action_count = 0
    total_actions = 0

    def __init__(self, width, height, window, color, objects, distance, x=None, y=None):
        super().__init__(width, height, window, color, distance, objects, x, y)

    def find_valid_location(self, objects, distance):
        not_in_range = True
        x, y = 0, 0
        while not_in_range:
            x, y = super().find_valid_location(objects, distance)
            if abs((WINDOW_X/2-PLAYER_DIM/2) - x) > curriculum.start_distance_x:
                continue
            elif abs((WINDOW_Y/2-PLAYER_DIM/2 - y)) > curriculum.start_distance_y:
                continue
            else:
                not_in_range = False
        return x, y

    def ai_move(self, ai_output, epsilon, current_time, player):

        if BIASED_EPSILON:
            epsilon = self.biased_epsilon(current_time, player, epsilon)
        else:
            self.normal_epsilon(current_time)

        if random.random() <= epsilon and EPSILON_ACTIVE and training:
            self.action = self.epsilon_action
        else:
            self.action = int(torch.argmax(ai_output).item())

        self.action_count += 1
        self.total_actions += self.action
        self.action_distribution.append(self.action)
        
        self.previous_x, self.previous_y = self.get_center()

        # Moving the AI using the action    
        directional_vector = self.get_directional_vector()
        if (directional_vector[0]):
            self.dy -= ACCELERATION
        if (directional_vector[1]):
            self.dy += ACCELERATION
        if (directional_vector[2]):
            self.dx -= ACCELERATION
        if (directional_vector[3]):
            self.dx += ACCELERATION

        self.apply_friction()
        self.check_max_velocity()        
        self.move()
        self.check_bounds()

    def biased_epsilon(self, current_time, player, epsilon):

        if not EPSILON_ACTIVE:
            return 0

        bias = 0

        player_x = self.moving_towards_player(player, "x")
        player_y = self.moving_towards_player(player, "y")
        wall_x = self.moving_into_wall("x")
        wall_y = self.moving_into_wall("y")
        if wall_x or wall_y:
            bias += 0.6

        if not(player_x or player_y):
            bias += 0.5

        # if not self.nearby_player(player):
        #     bias += 0.25

        if self.in_glue:
            bias += .4

        if self.collided_with_ai:
            bias += .5

        if self.dx == 0 and self.dy == 0:
            bias += .5

        if bias < 0:
            bias = 0

        multiplier = 1.0 + bias
        altered_epsilon = epsilon * multiplier
        altered_epsilon = max(0, min(1.0, altered_epsilon))

        if (bias > 0.7 and current_time - self.action_reset_timer >= self.action_reset_time) or self.epsilon_action is None:
            self.epsilon_action = random.randint(0, 8)
            self.action_reset_timer = current_time
            self.action_reset_time = random.randint(int(EPSILON_TIME_MIN/10), int((EPSILON_TIME_MAX/5) / (1.0 + bias)))

        return altered_epsilon
    
    def normal_epsilon(self, current_time):
        if not EPSILON_ACTIVE:
            return 0
        
        if current_time - self.action_reset_timer >= self.action_reset_time:
            self.epsilon_action = random.randint(0, 8)
            self.action_reset_timer = current_time
            self.action_reset_time = random.randint(EPSILON_TIME_MIN, EPSILON_TIME_MAX)

    def get_directional_vector(self):
        match self.action:
            case 0:
                return [False, False, False, False]
            case 1:
                return [True, False, False, False]
            case 2:
                return [False, True, False, False]
            case 3:
                return [False, False, True, False]
            case 4:
                return [False, False, False, True]
            case 5:
                return [True, False, True, False]
            case 6:
                return [True, False, False, True]
            case 7:
                return [False, True, True, False]
            case 8:
                return [False, True, False, True]
    
    def get_action_vector(self):
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        actions[self.action] = 1
        return actions

    def check_for_player_collisions(self, current_time, player):
        
        if self.hitbox.colliderect(player.hitbox):
            player.contacted_object = True
            player.objects.append(self)
            player.remove_objects_timer = current_time
            self.touching_player = True
        else:
            return

        # check for collision between the AI and player and removes 2 health from the player if they collide.
        if (current_time - self.timer >= DAMAGE_COOLDOWN):
            old_player_dx, old_player_dy = player.dx, player.dy
            player.dx = self.dx/2
            player.dy = self.dy/2
            self.dx = old_player_dx/2
            self.dy = old_player_dy/2
            self.timer = current_time
            player.health -= 2

    def check_for_ai_collisions(self, ai_list, current_time):
        # Checks for collisions between different AI objects and accounts for collision physics.

        for ai in ai_list:
            if ai == self:
                continue
            if self.hitbox.colliderect(ai.hitbox):
                
                self.collided_with_ai = True
                ai.collided_with_ai = True

                if (current_time - self.ai_collision_timer >= COLLISION_TIMER):
                    self.ai_collision_timer = current_time
                    old_ai_dx, old_ai_dy = ai.dx, ai.dy
                    ai.dx = self.dx/2
                    ai.dy = self.dy/2
                    self.dx = old_ai_dx/2
                    self.dy = old_ai_dy/2

    def moving_into_wall(self, axis='x'):
        # Checks if the AI is moving into a wall. If it is, it returns True.

        velocity = None
        location = None

        if axis == 'x':
            velocity = self.dx
            location = self.hitbox.x
            boundry = WINDOW_X
        elif axis == 'y':
            velocity = self.dy
            location = self.hitbox.y
            boundry = WINDOW_Y
        else:
            print("Error: The inputted axis is not valid. Please use 'x' or 'y'.")
            return False


        if velocity > 0 and location + PLAYER_DIM >= boundry - 10:
            return True
        elif velocity < 0 and location <= 10:
            return True
        elif location + PLAYER_DIM >= boundry:
            return True
        elif location <= 0:
            return True
        
        return False
    
    def moving_towards_player(self, player, axis='x'):
        # checks if the AI is moving towards the player.

        ai_location = None
        player_location = None
        velocity = None

        if axis == 'x':
            ai_location, _ = self.get_center()
            player_location, _ = player.get_center()
            velocity = self.dx
        elif axis == 'y':
            _, ai_location = self.get_center()
            _, player_location = player.get_center()
            velocity = self.dy
        else:
            return False

        moving_to_player = False
        if velocity > 0 and player_location > ai_location:
            moving_to_player = True
        elif velocity < 0 and player_location < ai_location:
            moving_to_player = True
        
        return moving_to_player
    
    def nearby_player(self, player):
        # Checks if the AI is close to the player.

        ai_x, ai_y = self.get_center()
        player_x, player_y = player.get_center()
        if ((player_x + PLAYER_DIM*2 > ai_x and player_x - PLAYER_DIM*2 < ai_x) 
            and (player_y + PLAYER_DIM*2 > ai_y and player_y - PLAYER_DIM*2 < ai_y)):
            return True
        else:
            return False


class TrainingData:
    # This class will store some of the training data for future training WOHOOOOOOO!!!
    def __init__(self, max_length):
        self.data = deque([], maxlen=max_length)

    def append(self, transition):
        # A transition is a tuple which contains information used for training the AI.
        # Transition = (AI memory, action made by AI, New memory after action was made, Reward earned from the action)
        self.data.append(transition)

    def get_sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def batch_data(self, batch_size):
        data = list(self.data)
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
        return batches
    
    def __len__(self):
        return len(self.data)


class HivemindLSTM(nn.Module):
    # We will be using an LSTM model in this experiment. This LSTM will control the AI objects.
    # How many Networks can a Network Neural if a Network could Neural Networks?

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
            h0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(device)
        if (c0 is None):
            c0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size).to(device)
        
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
    epsilon = INITIAL_EPSILON
    loss_fn = nn.SmoothL1Loss()
    resync_counter = 0
    previous_memory = None
    memory = torch.zeros((NUM_SAVED_FRAMES, SEQUENCE_LENGTH), dtype=torch.float32).to(device)

    glues = None
    player = None
    ai_list = None

    def __init__(self, num_ais, data_manager, model_number):
        # Set up the policy and target model.
        self.policy_model = HivemindLSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, num_ais).to(device)
        self.model_number = model_number
        self.load_model()
        self.target_model = HivemindLSTM(NUM_LAYERS, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE, num_ais).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=LEARNING_RATE)

        # Save data manager
        self.data_manager = data_manager

    def set(self, glues, player):
        self.glues = glues
        self.player = player
        self.ai_list = self.create_ais(NUM_AI_OBJECTS)

    def create_ais(self, num_ais):
        ais = []
        for i in range(num_ais):

            ai_object = AI(
                PLAYER_DIM, 
                PLAYER_DIM, 
                window, 
                RED,
                ([self.player] + self.glues),
                (PLAYER_DIM+GLUE_DIM)/2
                )
            ai_object.itr = i
            ais.append(ai_object)
        return ais
    
    def add_frame_to_memory(self, frame):
        frame = frame.to(device)

        for i in range(NUM_SAVED_FRAMES):
            if ((self.memory[NUM_SAVED_FRAMES - (i+1)] == torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32).to(device)).sum().item() == SEQUENCE_LENGTH):
                self.memory[NUM_SAVED_FRAMES - (i+1)] = frame
                return
        
        self.memory = torch.cat((self.memory[1 : NUM_SAVED_FRAMES], frame), dim=0)
    
    def update_memory(self):
        x, y = self.player.get_center()
        tensor = torch.tensor([
            x/WINDOW_X, 
            y/WINDOW_Y,
            self.player.dx/MAX_VELOCITY,
            self.player.dy/MAX_VELOCITY
            ], dtype=torch.float32)
        
        list_ai_info = []
        for ai in self.ai_list:
            x, y = ai.get_center()
            list_ai_info.append(x/WINDOW_X)
            list_ai_info.append(y/WINDOW_Y)
            list_ai_info.append(ai.dx/MAX_VELOCITY)
            list_ai_info.append(ai.dy/MAX_VELOCITY)
            list_ai_info.extend(ai.get_action_vector())
        ai_tensor = torch.tensor(list_ai_info, dtype=torch.float32)
        
        list_glues_location = []
        for glue in self.glues:
            x_glue, y_glue = glue.get_center()
            list_glues_location.append(x_glue/WINDOW_X)
            list_glues_location.append(y_glue/WINDOW_Y)


        glue_tensor = torch.tensor(list_glues_location, dtype=torch.float32)
        tensor = torch.cat((tensor, ai_tensor ,glue_tensor), dim=0)
        tensor = tensor.unsqueeze(0)
        self.add_frame_to_memory(tensor)

    def sync_models(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
    
    def move_ais(self, current_time):

        # Sync the Target model with the Policy models after 10 frames. 
        if self.resync_counter >= SYNC_MODEL: 
            self.sync_models()
            self.resync_counter = 0
        q_values, _, _ = self.policy_model(self.memory.unsqueeze(0))
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

        rewards = torch.zeros(len(self.ai_list))
        for i, ai in enumerate(self.ai_list):
            reward = 0

            # Pentalties AI for being in a glue.
            if ai.in_glue:
                reward += GLUE_CONTACT

            # Rewards AI for touching the player.
            if ai.touching_player:
                reward += PLAYER_CONTACT

            # Pentalties AI for moving into a wall (Like a bozo).
            if ai.moving_into_wall(axis='x') or ai.moving_into_wall(axis='y'):
                reward += WALL_CONTACT

            # Rewards AI for moving towards the player while not in a glue.
            if ai.moving_towards_player(self.player, axis='x') and not ai.in_glue:
                reward += MOVING_TOWARDS_PLAYER
            if ai.moving_towards_player(self.player, axis='y') and not ai.in_glue:
                reward += MOVING_TOWARDS_PLAYER

            # Checks for collisions between different AI objects and applies a negative reward if there is a collision.
            if ai.collided_with_ai:
                reward += AI_COLLISION
                ai.collided_with_ai = False

            # Pentalties AI for not moving.
            if round(ai.dx) == 0 and round(ai.dy) == 0:
                reward += NO_MOVEMENT

            ai_x, ai_y = ai.get_center()
            player_x, player_y = self.player.get_center()
            x_diff = ai_x - player_x
            y_diff = ai_y - player_y
            distance = math.sqrt(x_diff**2 + y_diff**2)  # PYTHAGOREAN THEOREM?!?!?!?! YIPPEE!!!!!!!!!!!!!!
            max_distance = ((WINDOW_X)**2 + (WINDOW_Y) **2)  # PYTHAGOREAN THEOREM AGAIN?!?!?!?!? AMAZING!!!!!!!!
            normalized_distance = distance/max_distance
            reward += math.sqrt(1-(2*normalized_distance))*DISTANCE_REWARD_MULTIPLIER


            # Rewards AI for closing the distance between itself and the player.
            # Pentalties AI for increasing the distance between itself and the player.
            if not ai.previous_x == None and not self.player.previous_x == None:

                # These values have a max of 2 and a min of -2 before being transformed to a range of 1 to -1
                x_difference = (abs(ai.previous_x - self.player.previous_x) - abs(x_diff))/MAX_VELOCITY
                y_difference = (abs(ai.previous_y - self.player.previous_y) - abs(y_diff))/MAX_VELOCITY

                # Now it will be transformed to a range of 1 to -1 and then averaged and added to the reward.
                reward += (x_difference * VELOCITY_REWARD_MULTIPLIER  + y_difference * VELOCITY_REWARD_MULTIPLIER)/2
                
            ai.in_glue = False
            rewards[i] = reward

        # Increases reward if multiple AIs are touching the player
        num_ais_touching_player = sum([1 for ai in self.ai_list if ai.touching_player])
        if num_ais_touching_player > 1 or len(self.ai_list) == 1:
            rewards += (num_ais_touching_player / len(self.ai_list))*.75
        
        rewards = torch.clamp(rewards, min=MIN_REWARDS, max=MAX_REWARDS)

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

    def train_ai(self, batch=None):
        
        if batch is None:
            batch = self.data_manager.get_sample(GAME_BATCH_SIZE)

        # Get actions and rewrads
        actions = torch.tensor([batch[i][1] for i in range(len(batch))], dtype=torch.long).to(device)
        rewards = torch.stack([batch[i][3] for i in range(len(batch))]).to(device)

        # Get the states and new states.
        states = [batch[i][0]for i in range(len(batch))]
        states = torch.stack(states, dim=0)
        new_states = [batch[i][2] for i in range(len(batch))]
        new_states = torch.stack(new_states, dim=0)

        q_values, _, _ = self.policy_model(states)

        with torch.no_grad():
            # Get max q values from policy model
            policy_next_q, _, _ = self.policy_model(new_states)
            policy_next_q_max = policy_next_q.argmax(dim=2)

            # Get Target evalutation
            target_next_q, _, _ = self.target_model(new_states)

            next_q_values = torch.zeros_like(rewards)
            for ai_idx in range(len(self.ai_list)):
                for batch_idx in range(len(batch)):
                    action_idx = policy_next_q_max[batch_idx, ai_idx]
                    next_q_values[batch_idx, ai_idx] = target_next_q[batch_idx, ai_idx, action_idx]

            # Calculate the Target using the split max q values. This will be used for training.
            targets = rewards + DISCOUNT_FACTOR * next_q_values

        total_loss = 0

        for ai_index in range(NUM_AI_OBJECTS):
            # Get individual Q_values for each AI object
            q_values_for_ai = q_values[:, ai_index, :]
            actions_for_ai = actions[:, ai_index]
            target_for_ai = targets[:, ai_index]

            # Match Q_values to the chosen action.
            q_values_to_action = torch.stack([q_values_for_ai[i][actions_for_ai[i]] for i in range(len(batch))])

            # Calculate loss using targets and individual AI q_values
            individual_loss = self.loss_fn(q_values_to_action, target_for_ai)
            total_loss += individual_loss

        # Average the total_loss.
        total_loss /= NUM_AI_OBJECTS

        # How does one determine if they are queer? Am I a "queer" individual? What is a queer?
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        self.total_loss += (total_loss.detach().to("cpu")).item()
        self.loss_count += 1
        return total_loss.item()

    def load_model(self):
        # This function checks if one of the saved models can be loaded, and if it can the function will load the model.
        # If not the function will create a new model and save its input shape so it can be saved and used in the future.

        # Check if model exists

        with open(INFO_FILE, 'r') as f:
            self.ai_save_data = json.load(f)
        
        empty, idx = self.match_model()
        if idx is None:
            print("Model not found. Creating a new one!")
            if empty:
                self.ai_save_data["Model"].append(1)
                self.model_number = 1
                idx = 0
            elif model_number is None:
                self.model_number = max(self.ai_save_data["Model"]) + 1
                idx = len(self.ai_save_data["Model"])
            else:
                idx = len(self.ai_save_data["Model"])
            self.ai_save_data["Model"].append(self.model_number)
            self.ai_save_data["Ais"].append(NUM_AI_OBJECTS)
            self.ai_save_data["Glues"].append(GLUES)
            self.ai_save_data["Hidden"].append(HIDDEN_SIZE)
            self.ai_save_data["Frames"].append(NUM_SAVED_FRAMES)
            self.ai_save_data["Layers"].append(NUM_LAYERS)
            self.ai_save_data["Epsilon"].append(INITIAL_EPSILON)
            self.idx = idx
            return
        self.idx = idx
        
        print(f"Found model {self.model_number}, Now loading model...")
        model_dir = SAVE_FOLDER + '/model_' + str(self.model_number) + ".pth"
        save_dict = torch.load(model_dir)
        self.policy_model.load_state_dict(save_dict)
        print(f"Model {self.model_number} successfully loaded!")
        
    def save_model(self):
        # Saves the model 

        with open(INFO_FILE, 'w') as f:
            json.dump(self.ai_save_data, f)

        print("Saving model")
        torch.save(self.policy_model.state_dict(), SAVE_FOLDER + "/" + "model_" + str(self.model_number) + ".pth")
        print(f"Model {self.model_number} saved successfully.\n")


    def match_model(self):

        if len(self.ai_save_data["Model"]) == 0:
            return True, None
            
        for i in range(len(self.ai_save_data["Model"])):
            ais = self.ai_save_data["Ais"][i] == NUM_AI_OBJECTS
            glues = self.ai_save_data["Glues"][i] == GLUES
            hidden = self.ai_save_data["Hidden"][i] == HIDDEN_SIZE
            frames = self.ai_save_data["Frames"][i] == NUM_SAVED_FRAMES
            layers = self.ai_save_data["Layers"][i] == NUM_LAYERS

            if ais and glues and hidden and frames and layers:
                self.model_number = self.ai_save_data["Model"][i]
                self.epsilon = self.ai_save_data["Epsilon"][i]
                idx = i
                return False, idx
                break

        return False ,None


class ProgressTracker:
    # Keeps track of important data to be displaied and saved them at the end.

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
        "Q Values Min": []
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
            "Q Values Min": [float(x) if hasattr(x, 'item') else x for x in self.data["Q Values Min"]]
        }
        
        file_name = DATA_FOLDER + "/" + "model_" + str(self.model_number) + "_" + str(self.iteration) +"_data.csv"
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

class CurriculumManager:
    # Things this will control:
    # player max velocity while random.
    # player's time spend fleeing from AI/reactions to AI
    # The amount of times the player moves
    # Movement of glues.
    player_max = 5
    player_reaction_time = 0
    player_movement = 0
    glue_movement = 0
    start_distance_x = 100
    start_distance_y = 100

    def __init__(self, training):
        if not training:
            self.player_max = MAX_VELOCITY
            self.player_reaction_time = PLAYER_CONTACT
            self.player_movement = 1
            self.glue_movement = 10
            self.start_distance_x = WINDOW_X
            self.start_distance_y = WINDOW_Y

    def increase_difficulty(self, episodes):
        if episodes >= TRAINING_INCREMENT:
            return
        self.player_max += (MAX_VELOCITY - 5)/TRAINING_INCREMENT
        self.player_reaction_time += PLAYER_CONTACT/TRAINING_INCREMENT
        self.player_movement += 1/TRAINING_INCREMENT
        self.glue_movement += 10/TRAINING_INCREMENT
        self.start_distance_x += (WINDOW_X - 100)/TRAINING_INCREMENT
        self.start_distance_y += (WINDOW_Y - 100)/TRAINING_INCREMENT


def kill_model():
    with open(INFO_FILE) as f:
        model_info = json.load(f)

        if len(model_info["Model"]) == 0:
            return None
        
        model_file = None

        for i in range(len(model_info["Model"])):
            ais = model_info["Ais"][i] == NUM_AI_OBJECTS
            glues = model_info["Glues"][i] == GLUES
            hidden = model_info["Hidden"][i] == HIDDEN_SIZE
            frames = model_info["Frames"][i] == NUM_SAVED_FRAMES
            layers = model_info["Layers"][i] == NUM_LAYERS

            if ais and glues and hidden and frames and layers:
                model_file = SAVE_FOLDER + '/model_' + str(model_info["Model"][i]) + ".pth"
                idx = i
                model_number = model_info["Model"][idx]
                print(f"Now Deleating Model {model_number}...")
                break

        if model_file is None:
            print("There is currently no model file saved. There is no model to be deleated.\n")
            return None

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
        with open(INFO_FILE, 'w') as f:
            json.dump(model_info, f)
            print("The model had been deleated successfully.\n")
        return model_number


def check_for_folder():
    # Creates AI save directory and JSON file if one is not present.

    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    if not os.path.isfile(INFO_FILE):
        open(INFO_FILE, "x")
        with open(INFO_FILE, "w") as f:
            json.dump(AI_SAVE_DATA, f)
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)


def display_training_text(text):
    window.fill(BLACK)
    top_text = training_font.render("Training AI:", True, WHITE)
    bottom_text = font.render(text, True, WHITE)

    top_rect = top_text.get_rect(center=(WINDOW_X/2, WINDOW_Y/2))
    bottom_rect = bottom_text.get_rect(center=(WINDOW_X/2, WINDOW_Y/2 + top_rect.height))

    window.blit(top_text, top_rect)
    window.blit(bottom_text, bottom_rect)
    pygame.display.flip()


def post_game_training(hivemind_manager, training_data):
    # Trains the AI after an episode ends when training=True
    data = training_data.batch_data(TRAINING_BATCH_SIZE)
    end = False
    total_time = 0
    bottom_text = "Beginning post game training..."
    print("\n" + bottom_text)
    display_training_text(bottom_text)

    for epoch in range(TRAINING_EPOCHS):

        beginning_time = pygame.time.get_ticks()
        loss = 0

        for i in range(len(data)):
            batch = data[i]
            loss += hivemind_manager.train_ai(batch=batch)
            if i % TRAINING_SYNC == 0:
                hivemind_manager.sync_models()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    end = True
                    break
            if end:
                break
        if end:
            print("Training cancelled")
            break

        loss /= len(data)
        ending_time = pygame.time.get_ticks()
        time = round(((ending_time-beginning_time)/1000), 3)
        total_time += time
        total_time = round(total_time, 3)
        print(f"Epoch {epoch + 1}/{TRAINING_EPOCHS} finished in {time} seconds")
        text = "Epoch: " + str(epoch) + "/" + str(TRAINING_EPOCHS) + " | Loss: " + str(round(loss, 5)) + " | Total Time: " + str(total_time)
        display_training_text(text)

    return end


def game_end(ai_manager, player, progress_tracker, time, data_manager, end):
    global previous_time

    if training and not end:
        end = post_game_training(ai_manager, data_manager)

    if EPSILON_ACTIVE and training:
        # Decay Epsilon
        new_epsilon = (ai_manager.epsilon - INITIAL_EPSILON/EPSILON_DECAY)
        if new_epsilon < 0:
            new_epsilon = 0
    else:
        new_epsilon = ai_manager.epsilon

    ai_manager.epsilon = new_epsilon
    if end:
        ai_manager.ai_save_data["Epsilon"][ai_manager.idx] = new_epsilon
        ai_manager.save_model()

    previous_time = pygame.time.get_ticks()

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
    time = round((TIME_LIMIT - time)/1000)
    for ai in ai_manager.ai_list:
        actions += ai.total_actions/ai.action_count
        spread += get_spread(ai.action_distribution, 0, 8)
        ai.action_distribution.clear()

    actions /= len(ai_manager.ai_list)
    spread /= len(ai_manager.ai_list)
    q_value_max = ai_manager.q_value_max/ai_manager.q_value_count
    q_value_average = ai_manager.q_value_average/ai_manager.q_value_count
    q_value_min = ai_manager.q_value_min/ai_manager.q_value_count

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
    
    return end


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


def draw_game(player, glues, time, ai_manager):
    # Draws the game on the window. Quite self explanatory.
    window.fill(BLACK)

    player.display()

    for ai in ai_manager.ai_list:
        ai.display()

    for glue in glues:
        glue.display()

    try: 
        text = font.render(f"Health: {player.health}  |  Time Remaining: {round((TIME_LIMIT - time)/1000)}  |  Episode: {episodes}/{MAX_EPISODES}", True, WHITE)
    except Exception:
        text = font.render("Health: 0", True, WHITE)
    window.blit(text, (10, 10))
    pygame.display.flip()


def main_loop(progress_tracker, data_manager, model_number, curriculum, ai_manager):
    end = False
    num_frames = 0
    running = True
    clock = pygame.time.Clock()
    player = Player(PLAYER_DIM, PLAYER_DIM, window, WHITE, objects=[], distance=None, curriculum=curriculum) # x=WINDOW_X/2-PLAYER_DIM/2, y=WINDOW_Y/2-PLAYER_DIM/2
    glues = []
    for _ in range(GLUES):
        glue = Glue(GLUE_DIM, GLUE_DIM, window, YELLOW, [player]+glues, distance=GLUE_DIM)
        glues.append(glue)
    ai_manager.set(glues, player)
    # ai_manager = AIHivemindManager(NUM_AI_OBJECTS, glues, player, data_manager, model_number)

    # Game loop, YIPPEEEEEEE
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
                end = game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end)
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    end = game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end)
                    running = False
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override

        if pygame.time.get_ticks()-previous_time >= TIME_LIMIT:
            end = game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end)
            running = False

        # AI moves first
        ai_manager.resync_counter += 1
        ai_manager.move_ais(current_time)
        for ai in ai_manager.ai_list:
            ai.check_for_player_collisions(current_time, player)

        for glue in glues:
            glue.check_for_collisions(ai_manager.ai_list + [player], current_time)
            glue.check_bounds()
            glue.apply_friction()
            if (current_time - glue.movement_timer >= GLUE_MOVEMENT_TIME):
                x, y = glue.random_move()
                glue.dx += round(curriculum.glue_movement) * x
                glue.dy += round(curriculum.glue_movement) * y
                glue.movement_timer = current_time
            glue.move()

        player.player_move(current_time)
        if player.health <= 0:
            end = game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end)
            running = False

        # Saved the changes made to the enviorment, Save the needed information for training, Then Train the AI.
        ai_manager.update_memory()
        ai_manager.save_data()
        if num_frames >= GAME_BATCH_SIZE:
            ai_manager.train_ai()

        draw_game(player, glues, current_time - previous_time, ai_manager)
        num_frames += 1
        clock.tick(60)
    
    return not end


if __name__ == "__main__":
    progress_tracker = ProgressTracker(iteration)
    data_manager = TrainingData(max_length=3600)
    curriculum = CurriculumManager(training)
    model_number = None
    if delete_model_file:
        model_number = kill_model()

    # player = Player(PLAYER_DIM, PLAYER_DIM, window, WHITE, objects=[], distance=None, curriculum=curriculum) # x=WINDOW_X/2-PLAYER_DIM/2, y=WINDOW_Y/2-PLAYER_DIM/2
    # glues = []
    # for _ in range(GLUES):
    #     glue = Glue(GLUE_DIM, GLUE_DIM, window, YELLOW, [player]+glues, distance=GLUE_DIM)
    #     glues.append(glue)
    ai_manager = AIHivemindManager(NUM_AI_OBJECTS, data_manager, model_number)

    episodes = 0
    run = True
    if MAX_EPISODES == 0:
        run = False
    check_for_folder()
    while run:
        progress_tracker.append(episodes, "Episodes")
        run = main_loop(progress_tracker, data_manager, model_number, curriculum, ai_manager)
        episodes += 1
        curriculum.increase_difficulty(episodes)
        if episodes > MAX_EPISODES:
            run = False
    
    pygame.quit()

    if training:

        print(f"Number of Episodes: {episodes}")
        try:
            progress_tracker.save_as_cvs()
            print("Pls")
        except:
            data = progress_tracker.data
            print(f"Episodes: {data["Episodes"][0]} | Rewards: {data["Rewards"][0]} | Actions: {data["Actions"][0]}\n")
            print(f"Health: {data['Health'][0]} | Loss: {data['Loss'][0]} | Epsilon: {data['Epsilon'][0]} | Time: {data['Time'][0]}")

        progress_tracker.graph_results()
