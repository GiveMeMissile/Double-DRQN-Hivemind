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
#   Iter 31: eps: 150, ais: 1, glues: 5, lr: 0.0001, Discount: 0.9. Other Changes: Added temporal Learning (took 4 hours lol)
#   Iter 32: eps: 150, ais: 1, glues: 5, lr: 0.00001, Discount: 0.9, Other Changes: Reverted max batch size 16 --> 64
#   Iter 33: eps: 150, ais: 1, glues: 5, lr: 0.0005, Discount: 0.9, Other Changes: Added training data saving
#   Iter 33.5: eps: 150, ais: 1, glues: 4, lr: 0.00005, Discount: 0.9, Other Changes: Organized files (Testing to see if everything works well LOL).
#   Note: New Iteration Policy: 2/3 episodes data collection and training, 1/3 episode after using data for deep training.
#   Iter 34: pre eps: 100, post eps: 50, ais: 1, glues: 0, lr: 0.0001, Discount: 0.9, Other Changes: Fixed collision tracking bug and changed biased epsilon.
#   Iter 34.5: pre eps: 100, post eps: 50, ais: 1, glues: 0, lr: 0.00001, Discount: 0.9, Other Changes: Fixed data saving errors seen in iter 34

#####
# Currently working on getting training to work.
#####

import pygame
import utils
import Config as init
from DataManager import ProgressTracker, SequentialTrainingData
from Objects import Player, Glue
from DDRQN import AIHivemindManager
from Curriculum import CurriculumManager

def game_loop(progress_tracker, data_manager, curriculum, ai_manager):
    data_manager.model_number = ai_manager.model_number
    end = False
    num_frames = 0
    running = True
    clock = pygame.time.Clock()
    player = Player(
        init.PLAYER_DIM, 
        init.PLAYER_DIM, 
        window,
        init.WHITE,
        init.PLAYER_DIM,
        curriculum
    ) # x=WINDOW_X/2-PLAYER_DIM/2, y=WINDOW_Y/2-PLAYER_DIM/2
    glues = []
    for _ in range(init.GLUES):
        glue = Glue(
            init.GLUE_DIM, 
            init.GLUE_DIM, 
            window, 
            init.YELLOW,
            distance=init.GLUE_DIM,
            objects=[player]+glues
        )
        glues.append(glue)
    ai_manager.set(glues, player, init.PLAYER_DIM, init.RED)

    # Game loop, YIPPEEEEEEE
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
                end, previous_time_new = utils.game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end, episodes, window)
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset the game if the space key is pressed.
                    end, previous_time_new = utils.game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end, episodes, window)
                    running = False
                elif event.key == pygame.K_LSHIFT:
                    player.override = not player.override

        if pygame.time.get_ticks()-previous_time >= init.TIME_LIMIT:
            end, previous_time_new = utils.game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end, episodes, window)
            running = False

        # AI moves first
        ai_manager.resync_counter += 1
        ai_manager.move_ais(current_time, init.SYNC_MODEL)
        for ai in ai_manager.ai_list:
            ai.check_for_player_collisions(current_time, player)

        for glue in glues:
            glue.check_for_collisions(ai_manager.ai_list + [player], current_time)
            glue.check_bounds()
            glue.apply_friction()
            if (current_time - glue.movement_timer >= init.GLUE_MOVEMENT_TIME):
                x, y = glue.random_move()
                glue.dx += round(curriculum.glue_movement) * x
                glue.dy += round(curriculum.glue_movement) * y
                glue.movement_timer = current_time
            glue.move()

        player.player_move(current_time)
        if player.health <= 0:
            end, previous_time_new = utils.game_end(ai_manager, player, progress_tracker, current_time - previous_time, data_manager, end, episodes, window)
            running = False

        # Saved the changes made to the enviorment, Save the needed information for training, Then Train the AI.
        ai_manager.update_memory()
        ai_manager.save_data()
        # if num_frames >= GAME_BATCH_SIZE:
        #     ai_manager.train_ai()

        utils.draw_game(player, glues, current_time - previous_time, ai_manager, episodes, window)
        num_frames += 1
        clock.tick(60)
    
    return not end, previous_time_new


if __name__ == "__main__":
    window = pygame.display.set_mode((init.WINDOW_X, init.WINDOW_Y))
    pygame.init()
    previous_time = 0
    utils.check_for_folder()
    progress_tracker = ProgressTracker(init.iteration)
    data_manager = SequentialTrainingData(init.TRAINING_DATA_FOLDER, init.TRAINING_DATA_INFO, max_episodes=init.MAX_EPISODES)
    if init.training and init.LOAD_PREVIOUS_DATA:
        print("Loading Data...")
        data_manager.load_data(num_episodes=50)
        print("Data Loaded!!!")
    curriculum = CurriculumManager(init.training, init.MAX_VELOCITY, init.PLAYER_CONTACT, init.WINDOW_X, init.WINDOW_Y, init.TRAINING_INCREMENT)
    model_number = None
    if init.delete_model_file:
        model_number = utils.kill_model()

    ai_manager = AIHivemindManager(
        init.NUM_AI_OBJECTS, 
        data_manager, 
        model_number, 
        init.NUM_LAYERS, 
        init.HIDDEN_SIZE, 
        init.OUTPUT_SIZE, 
        init.LEARNING_RATE,
        curriculum,
        window,
        init.NUM_SAVED_FRAMES,
        init.SEQUENCE_LENGTH
    )

    episodes = 1
    run = True
    if init.MAX_EPISODES == 0:
        run = False
    while run:
        progress_tracker.append(episodes, "Episodes")
        run, previous_time = game_loop(progress_tracker, data_manager, curriculum, ai_manager)
        episodes += 1
        curriculum.increase_difficulty(episodes)
        if episodes > init.MAX_EPISODES:
            run = False
    
    new_epsilon = (ai_manager.epsilon - init.INITIAL_EPSILON/init.EPSILON_DECAY)
    if new_epsilon < 0:
        new_epsilon = 0
    ai_manager.ai_save_data["Epsilon"][ai_manager.idx] = new_epsilon
    ai_manager.save_model(init.INFO_FILE, init.SAVE_FOLDER)
    
    pygame.quit()
    if init.DATA_COLLECTION:
        data_manager.save_data(episodes)

    if init.training and init.PROGRESS_DATA_SAVE:

        print(f"Number of Episodes: {episodes}")
        try:
            progress_tracker.save_as_cvs()
            print("Pls")
        except:
            data = progress_tracker.data
            print(f"Episodes: {data["Episodes"][0]} | Rewards: {data["Rewards"][0]} | Actions: {data["Actions"][0]}\n")
            print(f"Health: {data['Health'][0]} | Loss: {data['Loss'][0]} | Epsilon: {data['Epsilon'][0]} | Time: {data['Time'][0]}")

        progress_tracker.graph_results()