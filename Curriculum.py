#######################################################################################################
# File Name:
#   CurriculumManager
#
# Purpose:
#   Contains CurriculumManager class which manages the
#   difficulty of the enviorment for the AI during training
#
# Used:
#   1 - Used by main to manage difficulity, I cannot believe I created an entire file for this one class...
#######################################################################################################

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

    def __init__(self, training, curriculum, max_veloicty, player_contact, win_x, win_y, training_increment):
        self.training_increment = training_increment
        self.max_velocity = max_veloicty
        self.player_contact = player_contact
        self.window_x = win_x
        self.window_y = win_y
        if not training or not curriculum:
            self.player_max = max_veloicty
            self.player_reaction_time = player_contact
            self.player_movement = 1
            self.glue_movement = 10
            self.start_distance_x = win_x
            self.start_distance_y = win_y

    def increase_difficulty(self, episodes):
        # Function which alters values within the game enviorment to make thingd harder for the AI

        if episodes >= self.training_increment:
            return
        self.player_max += (self.max_velocity - 5)/self.training_increment
        self.player_reaction_time += self.player_contact/self.training_increment
        self.player_movement += 1/self.training_increment
        self.glue_movement += 10/self.training_increment
        self.start_distance_x += (self.window_x - 100)/self.training_increment
        self.start_distance_y += (self.window_y - 100)/self.training_increment