#######################################################################################################
# File Name:
#   Objects
#
# Purpose:
#   Contains all of the game Objects which interact
#   with each other within the enviorment. 
#
# Used:
#   1 - Used by main to create glues and player.
#   2 - Used by DDRQN to create AI objects in AIManagerHivemind
#######################################################################################################

import torch
import pygame
import random
import Config as init

class Object:
    # Base class for all objects in the game. It has a hitbox, velocity, color, and more ig.

    def __init__(self, width, height, window, color, distance, objects=None, x=None, y=None):
        self.width = width
        self.height = height
        self.dx = 0
        self.dy = 0
        self.color = color
        self.window = window
        self.window_x = init.WINDOW_X
        self.window_y = init.WINDOW_Y
        self.friction = init.FRICTION
        self.max_velocity = init.MAX_VELOCITY
        self.acceleration = init.ACCELERATION
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
        elif self.hitbox.x > self.window_x - self.width:
            self.hitbox.x = self.window_x - self.width
            self.dx = -self.dx/2

        if self.hitbox.y < 0:
            self.hitbox.y = 0
            self.dy = -self.dy/2
        elif self.hitbox.y > self.window_y - self.height:
            self.hitbox.y = self.window_y - self.height
            self.dy = -self.dy/2

        if self.hitbox.x < -25:
            self.hitbox.x += 100
        elif self.hitbox.x > self.window_x + 25:
            self.hitbox.x -= 100

        if self.hitbox.y < -25:
            self.hitbox.y += 100
        elif self.hitbox.y > self.window_y + 25:
            self.hitbox.y -= 100

    def random_move(self):
        # Selects a random direction for the object to move in.
        x = random.randint(-1, 1)
        y = random.randint(-1, 1)
        return x, y

    def apply_friction(self):

        if self.dx > 0:
            self.dx -= self.friction
        elif self.dx < 0:
            self.dx += self.friction
        if self.dy > 0:
            self.dy -= self.friction
        elif self.dy < 0:
            self.dy += self.friction
        
        if self.dx < self.friction and self.dx > -self.friction:
            self.dx = 0
        if self.dy < self.friction and self.dy > -self.friction:
            self.dy = 0
            
    
    def check_max_velocity(self):
        if self.dx > self.max_velocity:
            self.dx = self.max_velocity
        elif self.dx < -self.max_velocity:
            self.dx = -self.max_velocity
        if self.dy > self.max_velocity:
            self.dy = self.max_velocity
        elif self.dy < -self.max_velocity:
            self.dy = -self.max_velocity

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
            x = random.randint(0, self.window_x - int(self.width/2))
            y = random.randint(0, self.window_y - int(self.height/2))
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

    def __init__(self, width, height, window, color, distance, objects=None, x=None, y=None):
        super().__init__(width, height, window, color, distance, objects=objects, x=x, y=y)
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
                if current_time - self.timer >= init.DAMAGE_COOLDOWN:
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

    def __init__(self, width, height, window, color, distance, curriculum, objects=None, x=None, y=None):
        super().__init__(width, height, window, color, distance, objects=objects, x=x, y=y)
        self.override = not init.RANDOM_MOVE
        self.curriculum = curriculum

    def player_move(self, current_time):
        # Player movement using WASD keys. The player can move in all directions and has a maximum velocity.

        keys = pygame.key.get_pressed()
        # Checking for key clicks and adding the proper acceleration to the velocity.

        self.previous_x, self.previous_y = self.get_center()

        if keys[pygame.K_w]:
            self.dy -= self.acceleration
            self.override = True
        if keys[pygame.K_s]:
            self.dy += self.acceleration
            self.override = True
        if keys[pygame.K_a]:
            self.dx -= self.acceleration
            self.override = True
        if keys[pygame.K_d]:
            self.dx += self.acceleration
            self.override = True
        
        self.apply_friction()
        if not init.training or self.override:
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
            if max > self.max_velocity:
                self.curriculum.player_max = self.max_velocity

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
            self.change_direction_time = random.randint(init.PLAYER_TIME_MIN, init.PLAYER_TIME_MAX)
        if not self.contacted_object:
            if self.curriculum.player_movement < random.random():
                self.move()
                self.check_bounds()
                return
            self.dx += self.acceleration * self.x_direction
            self.dy += self.acceleration * self.y_direction
            self.move()
            self.check_bounds()
            return
        
        # If there is a collision with 1 or more objects then the player object will accelerate away from the object(s) which collided with it.
        # This allows for better automated movement of the player square when it is not being controlled by the player.
        avg_x, avg_y = self.get_objects_average()
        x, y = self.location()
        if avg_x < x:
            self.dx += self.acceleration
        else:
            self.dx -= self.acceleration

        if avg_y < y:
            self.dy += self.acceleration
        else:
            self.dy -= self.acceleration

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

    def __init__(self, width, height, window, color, distance, curriculum, objects=None, x=None, y=None):
        self.curriculum = curriculum
        super().__init__(width, height, window, color, distance, objects=objects, x=x, y=y)

    def find_valid_location(self, objects, distance):
        not_in_range = True
        x, y = 0, 0
        while not_in_range:
            x, y = super().find_valid_location(objects, distance)
            if abs((self.window_x/2-init.PLAYER_DIM/2) - x) > self.curriculum.start_distance_x:
                continue
            elif abs((self.window_y/2-init.PLAYER_DIM/2 - y)) > self.curriculum.start_distance_y:
                continue
            else:
                not_in_range = False
        return x, y

    def ai_move(self, ai_output, epsilon, current_time, player):

        if init.BIASED_EPSILON:
            epsilon = self.biased_epsilon(current_time, player, epsilon)
        else:
            self.normal_epsilon(current_time)

        if random.random() <= epsilon and init.EPSILON_ACTIVE:
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
            self.dy -= self.acceleration
        if (directional_vector[1]):
            self.dy += self.acceleration
        if (directional_vector[2]):
            self.dx -= self.acceleration
        if (directional_vector[3]):
            self.dx += self.acceleration

        self.apply_friction()
        self.check_max_velocity()        
        self.move()
        self.check_bounds()

    def biased_epsilon(self, current_time, player, epsilon):

        if not init.EPSILON_ACTIVE:
            return 0

        bias = 1

        player_x = self.moving_towards_player(player, "x")
        player_y = self.moving_towards_player(player, "y")
        wall_x = self.moving_into_wall("x")
        wall_y = self.moving_into_wall("y")
        if wall_x or wall_y:
            bias -= 0.9

        if (player_x or player_y):
            bias += 2

        if player_x and player_y:
            bias += 3

        # if not self.nearby_player(player):
        #     bias -= 0.25

        if self.touching_player:
            bias += 5

        if self.in_glue:
            bias -= .7

        if self.collided_with_ai:
            bias -= 1

        if self.dx == 0 and self.dy == 0:
            bias -= .8

        if bias < 0:
            bias = 0

        altered_epsilon = epsilon * bias
        altered_epsilon = max(0, min(1.0, altered_epsilon))

        if (current_time - self.action_reset_timer >= self.action_reset_time * bias) or self.epsilon_action is None:
            self.epsilon_action = random.randint(0, 8)
            self.action_reset_timer = current_time
            self.action_reset_time = random.randint((init.EPSILON_TIME_MIN), init.EPSILON_TIME_MAX)

        return altered_epsilon
    
    def normal_epsilon(self, current_time):
        if not init.EPSILON_ACTIVE:
            return 0
        
        if current_time - self.action_reset_timer >= self.action_reset_time:
            self.epsilon_action = random.randint(0, 8)
            self.action_reset_timer = current_time
            self.action_reset_time = random.randint(init.EPSILON_TIME_MIN, init.EPSILON_TIME_MAX)

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
            self.touching_player = False
            return

        # check for collision between the AI and player and removes 2 health from the player if they collide.
        if (current_time - self.timer >= init.DAMAGE_COOLDOWN):
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

                if (current_time - self.ai_collision_timer >= init.COLLISION_TIMER):
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
            boundry = self.window_x
        elif axis == 'y':
            velocity = self.dy
            location = self.hitbox.y
            boundry = self.window_y
        else:
            print("Error: The inputted axis is not valid. Please use 'x' or 'y'.")
            return False


        if velocity > 0 and location + init.PLAYER_DIM >= boundry - 10:
            return True
        elif velocity < 0 and location <= 10:
            return True
        elif location + init.PLAYER_DIM >= boundry:
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
        if ((player_x + init.PLAYER_DIM*2 > ai_x and player_x - init.PLAYER_DIM*2 < ai_x) 
            and (player_y + init.PLAYER_DIM*2 > ai_y and player_y - init.PLAYER_DIM*2 < ai_y)):
            return True
        else:
            return False