#######################################################################################################
# File Name: 
#   RewardCalc
# 
# Purpose:
#   To utilize saved tensor data in order to calculate
#   new reward values using new reward constants
# 
# Used: 
#   1 - In Training when config reward constants != training reward constants
#######################################################################################################

import math
import torch
import Config as init

class RewardCalculator:

    ais = None
    glues = None
    window_x = None
    window_y = None

    def __init__(
            self, 
            min_reward, 
            max_reward, 
            velocity_reward, 
            distance_reward,
            player_reward,
            glue_reward,
            wall_reward,
            movement_reward,
            no_movement_reward,
            ai_reward
        ):
            self.min_reward = min_reward
            self.max_reward = max_reward
            self.velocity_reward = velocity_reward
            self.distance_reward = distance_reward
            self.player_reward = player_reward
            self.glue_reward = glue_reward
            self.wall_reward = wall_reward
            self.movement_reward = movement_reward
            self.no_movement_reward = no_movement_reward
            self.ai_reward = ai_reward


    def calculate_rewards(self, glues, ais, player):

        rewards = torch.zeros(self.ais)

        for i, ai in enumerate(ais):
            reward = 0

            in_glue = False

            for glue in glues:
                if self.in_glue(ai, glue):
                    reward += self.glue_reward
                    in_glue = True
                    break
            
            if self.in_player(ai, player):
                reward += self.player_reward

            if self.wall_contact(ai):
                reward += self.wall_reward

            if self.moving_towards_player(ai, player, True) and not in_glue:
                reward += self.movement_reward
            if self.moving_towards_player(ai, player, False) and not in_glue:
                reward += self.movement_reward

            for j, other_ai in enumerate(ais):
                if j == i:
                    continue
                if (self.ai_contact(ai, other_ai)):
                    reward += self.ai_reward
                    break
            
            if self.no_movement(ai):
                reward += self.no_movement_reward

            reward += self.calculate_distance_reward(ai, player)
            reward += self.calculate_difference_rewards(ai, player)

            rewards[i] = reward
        
        rewards = torch.clamp(rewards, min=self.min_reward, max=self.max_reward)
        return rewards

    def calculate_difference_rewards(self, ai, player):
        ai_x, ai_y, ai_old_x, ai_old_y = ai[0], ai[1], ai[4], ai[5]
        px, py, px_old, py_old = player[0], player[1], player[4], player[5]
        if ai_old_x == 0 and ai_old_y == 0 and px_old == 0 and py_old == 0:
            return 0
        diff_x = (abs(ai_old_x - px_old) - abs(ai_x - px))/init.MAX_VELOCITY
        diff_y = (abs(ai_old_y - py_old) - abs(ai_y - py))/init.MAX_VELOCITY
        return (diff_x * self.velocity_reward + diff_y * self.velocity_reward)/2

    def calculate_distance_reward(self, ai, player):
        # Calculates and returns the distance reward.

        ai_x, ai_y = ai[0], ai[1]
        px, py = player[0], player[1]
        x_diff = ai_x - px
        y_diff = ai_y - py
        distance = math.sqrt(x_diff**2 + y_diff**2)
        max_distance = math.sqrt((self.window_x)**2 + (self.window_y)**2)
        normalized_distance = distance/max_distance
        return math.sqrt(1-(normalized_distance))*self.distance_reward
    
    def in_player(self, ai, player):
        # Returns if the AI is touching the player (True or False)

        d = init.PLAYER_DIM/2
        ai_x, ai_y = ai[0], ai[1]
        px, py = player[0], player[1]
        if not (ai_x + d >= px - d and ai_x - d <= px + d):
            return False
        if not (ai_y + d >= py - d and ai_y - d <= py + d):
            return False
        return True
    
    def in_glue(self, ai, glue):
        da = init.PLAYER_DIM/2
        dg = init.GLUE_DIM/2
        ai_x, ai_y = ai[0], ai[1]
        gx, gy = glue[0], glue[1]

        if not(ai_x + da > gx - dg and ai_x - da < gx + dg):
            return False
        if not(ai_y + da > gy - dg and ai_y - da < gy + dg):
            return False
        return True
    
    def wall_contact(self, ai):
        # Determines if an ai is touching a wall of the enviorment 
        x, y = ai[0], ai[1]
        dx, dy = ai[2], ai[3]
        d = init.PLAYER_DIM
        x, y = round(x - d/2, 4), round(y - d/2, 4)

        # Check x
        if dx > 0 and x + d >= self.window_x - 10:
            return True
        elif dx < 0 and x <= 10:
            return True
        elif x + d >= self.window_x:
            return True
        elif x <= 0:
            return True
        
        # Check y
        if dy > 0 and y + d >= self.window_y - 10:
            return True
        elif dy < 0 and y <= 10:
            return True
        elif y + d >= self.window_y:
            return True
        elif y <= 0:
            return True
        
        return False

    def moving_towards_player(self, ai, player, x):
        if x:
            ai_l = ai[0]
            pl = player[0]
            ai_v = ai[2]
        else:
            ai_l = ai[1]
            pl = player[1]
            ai_v = ai[3]
        
        d = init.PLAYER_DIM/2
        if ai_v > 0 and pl > ai_l:
            return True
        if ai_v < 0 and pl < ai_l:
            return True
        if (ai_l - d < pl + d and ai_l + d > pl - d):
            return True
        
        return False


    def no_movement(self, ai):
        if round(ai[2]) == 0 and round(ai[3]) == 0:
            return True
        return False
    
    def ai_contact(self, ai_1, ai_2):
        ai_1x, ai_1y = ai_1[0], ai_1[1]
        ai_2x, ai_2y = ai_2[0], ai_2[1]
        d = init.PLAYER_DIM/2

        if not (ai_1x + d > ai_2x - d and ai_1x - d < ai_2x + d):
            return False
        if not (ai_1y + d > ai_2y - d and ai_1y - d < ai_2y + d):
            return False
        return True

    def denormalize_tensor_data(self, tensor, metadata, metadata_idx, prev_tensor):
        idx = 4
        win_x = metadata["Window X"][metadata_idx]
        win_y = metadata["Window Y"][metadata_idx]
        glues = metadata["Glues"][metadata_idx]
        ais = metadata["Ais"][metadata_idx]

        self.window_x = win_x
        self.window_y = win_y
        self.glues = glues
        self.ais = ais

        player_data = [
            round(tensor[0].item() * win_x),
            round(tensor[1].item() * win_y),
            round(tensor[2].item() * init.MAX_VELOCITY, 1), 
            round(tensor[3].item() * init.MAX_VELOCITY, 1),
            round(prev_tensor[0].item() * win_x),
            round(prev_tensor[1].item() * win_y)
        ]

        ai_data = []
        for _ in range(ais):
            ai = []
            ai.append(round(tensor[idx].item() * win_x))
            ai.append(round(tensor[idx + 1].item() * win_y))
            ai.append(round(tensor[idx + 2].item() * init.MAX_VELOCITY, 1))
            ai.append(round(tensor[idx + 3].item() * init.MAX_VELOCITY, 1))
            ai.append(round(prev_tensor[idx].item() * win_x))
            ai.append(round(prev_tensor[idx + 1].item() * win_y))
            ai_data.append(ai)
            idx += 13

        glue_data = []
        for _ in range(glues):
            glue = []
            glue.append(round(tensor[idx].item() * win_x))
            glue.append(round(tensor[idx + 1].item() * win_y))
            glue_data.append(glue)
            idx += 2

        return player_data, ai_data, glue_data