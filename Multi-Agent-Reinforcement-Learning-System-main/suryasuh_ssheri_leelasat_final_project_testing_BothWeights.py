import pygame
import torch
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import deque
import pygame
import cv2
from enum import Enum
import math
from collections import namedtuple
from gym import spaces
import matplotlib.pyplot as plt
from IPython import display
import torch.nn.functional as F

pygame.init()
font = pygame.font.Font('./Assets/arial.ttf', 25)

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

def manhattan_distance(p1, p2):
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_shape)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # print(F.softmax(self.fc_actor(x), dim=-1))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        return policy

actor_net = ActorNetwork(24, 4)
# Load the trained models
Q_net_path = "./Checkpoints/Shaktiman_DQN_CheckPoints/Grid_Q_net_1990.pkl"
actor_net_path = "./Checkpoints/Shaktiman_DQN_CheckPoints/actor_net_1990.pkl"

with open(Q_net_path, "rb") as f:
    Q_net = pickle.load(f)

with open(actor_net_path, "rb") as f:
    actor_net = pickle.load(f)

# Set the models to evaluation mode
Q_net.eval()
actor_net.eval()
BLOCK_SIZE = 40

class ShaktimanEnv:
    def __init__(self, speed, num_health_potions):
        self.w = 880
        self.h = 720
        self.num_health_potions = num_health_potions
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Shaktiman')
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load('./Assets/city.png').convert_alpha()
        self.background = pygame.transform.scale(self.background, (self.w, self.h))
        self.wall_img = pygame.image.load("./Assets/wall.jpg").convert_alpha()
        self.wall_img = pygame.transform.scale(self.wall_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.hero_img = pygame.image.load('./Assets/hero.png').convert_alpha()
        self.hero_img = pygame.transform.scale(self.hero_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.heman_img = pygame.image.load('./Assets/heman.png').convert_alpha()
        self.heman_img = pygame.transform.scale(self.heman_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.enemy_img = pygame.image.load('./Assets/enemy.png').convert_alpha()
        self.enemy_img = pygame.transform.scale(self.enemy_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.hp_img = pygame.image.load('./Assets/health.png').convert_alpha()
        self.hp_img = pygame.transform.scale(self.hp_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.reset()
        self.max_steps = 1000
        self.rew = 0
        self.speed = speed
        self.hero_health = 100
        self.heman_health = 100
        self.health_potions = []
        self.hero_score = 0
        self.heman_score = 0
        self.tot_hero_score = 0 
        self.tot_heman_score = 0 

        self.observation_space = spaces.Discrete(len(self.maze) * len(self.maze[0]))
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.maze = self.generate_maze()
        # self.hero = {'position': Point(BLOCK_SIZE, BLOCK_SIZE)}
        # self.heman = {'position': Point(BLOCK_SIZE * 2, BLOCK_SIZE * 2)}
        self.health_potions = []
        self.place_health_potions()
        self.enemies = self.place_enemies()
        self.score = 0
        self.enemies_caught = [False] * 5
        self.current_step = 0
        self.hero_health = 100
        self.heman_health = 100
        self.enemy_move_counter = 0
        self.hero_score = 0
        self.heman_score = 0
        self.hero = {'position': Point(BLOCK_SIZE, BLOCK_SIZE), 'direction': 0}  # Initialize hero direction to 0 (left)
        self.heman = {'position': Point(BLOCK_SIZE * 2, BLOCK_SIZE * 2), 'direction': 1}  # Initialize Heman direction to 1 (right)

        # print("hey heye")
        # print(self.get_observation())
        return self.get_observation()

    def generate_maze(self):
        maze = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]

        return maze

    # def generate_maze(self):
    #     maze = [
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #    ]

    #     return maze
    
    def place_enemies(self):
       enemies = []
       enemy_positions = [(800, 160), (160, 160), (160, 560), (800, 560), (480, 360)]
       for x, y in enemy_positions:
           enemies.append({'position': Point(x, y), 'caught': False})
       return enemies

    def place_health_potions(self):
        for _ in range(self.num_health_potions):
            while True:
                x = random.randint(1, len(self.maze[0]) - 2) * BLOCK_SIZE
                y = random.randint(1, len(self.maze) - 2) * BLOCK_SIZE
                if self.maze[y // BLOCK_SIZE][x // BLOCK_SIZE] == 0 and self.far_from_objects(x, y):
                    self.health_potions.append({'position': Point(x, y), 'found': False})
                    break

    def far_from_objects(self, x, y):
        if hasattr(self, 'enemy'):
        # for enemy in self.enemies:
            dist = math.sqrt((self.enemy['position'].x - x) ** 2 + (self.enemy['position'].y - y) ** 2)
            if dist < BLOCK_SIZE * 2:
                return False
        for obj in self.health_potions:
            dist = math.sqrt((obj['position'].x - x) ** 2 + (obj['position'].y - y) ** 2)
            if dist < BLOCK_SIZE * 2:
                return False
        return True

    # def get_observation(self):
    #     hero_x, hero_y = self.hero['position']
    #     hero_x //= BLOCK_SIZE
    #     hero_y //= BLOCK_SIZE
    #     heman_x, heman_y = self.heman['position']
    #     heman_x //= BLOCK_SIZE
    #     heman_y //= BLOCK_SIZE
    #     observation = [0] * (len(self.maze) * len(self.maze[0]))

    #     idx = hero_y * len(self.maze[0]) + hero_x
    #     observation[idx] = 1

    #     idx = heman_y * len(self.maze[0]) + heman_x
    #     observation[idx] = 0.5

    #     for enemy in self.enemies:
    #         if not enemy['caught']:
    #             enemy_x, enemy_y = enemy['position']
    #             enemy_x //= BLOCK_SIZE
    #             enemy_y //= BLOCK_SIZE
    #             idx = enemy_y * len(self.maze[0]) + enemy_x
    #             observation[idx] = 0.25

    #     return observation
    def get_observation(self):
        hero_observation = self.get_observation_hero()
        heman_observation = self.get_observation_heman()
        observation = hero_observation + heman_observation
        return observation

    def get_observation_hero(self):
        # Initialize the observation vector
        observation = [0] * 12

        # Get the current position and direction of the hero
        hero_x, hero_y = self.hero['position']
        hero_x //= BLOCK_SIZE
        hero_y //= BLOCK_SIZE
        hero_direction = self.hero['direction']

        # Check for danger in the left, right, up, and down directions
        if hero_x > 0 and self.maze[hero_y][hero_x - 1] == 1:
            observation[0] = 1  # Danger on the left
        if hero_x < len(self.maze[0]) - 1 and self.maze[hero_y][hero_x + 1] == 1:
            observation[1] = 1  # Danger on the right
        if hero_y > 0 and self.maze[hero_y - 1][hero_x] == 1:
            observation[2] = 1  # Danger on the up
        if hero_y < len(self.maze) - 1 and self.maze[hero_y + 1][hero_x] == 1:
            observation[3] = 1  # Danger on the down

        # Indicate the current direction of the hero
        if hero_direction == 0:  # Left
            observation[4] = 1
        elif hero_direction == 1:  # Right
            observation[5] = 1
        elif hero_direction == 2:  # Up
            observation[6] = 1
        elif hero_direction == 3:  # Down
            observation[7] = 1

        # Check for enemies in the left, right, up, and down directions
        for enemy in self.enemies:
            if not enemy['caught']:
                enemy_x, enemy_y = enemy['position']
                enemy_x //= BLOCK_SIZE
                enemy_y //= BLOCK_SIZE
                if enemy_x < hero_x:
                    observation[8] = 1  # Enemy on the left
                elif enemy_x > hero_x:
                    observation[9] = 1  # Enemy on the right
                if enemy_y < hero_y:
                    observation[10] = 1  # Enemy on the up
                elif enemy_y > hero_y:
                    observation[11] = 1  # Enemy on the down

        return observation

    def get_observation_heman(self):
        # Initialize the observation vector
        observation = [0] * 12

        # Get the current position and direction of Heman
        heman_x, heman_y = self.heman['position']
        heman_x //= BLOCK_SIZE
        heman_y //= BLOCK_SIZE
        heman_direction = self.heman['direction']

        # Check for danger in the left, right, up, and down directions
        if heman_x > 0 and self.maze[heman_y][heman_x - 1] == 1:
            observation[0] = 1  # Danger on the left
        if heman_x < len(self.maze[0]) - 1 and self.maze[heman_y][heman_x + 1] == 1:
            observation[1] = 1  # Danger on the right
        if heman_y > 0 and self.maze[heman_y - 1][heman_x] == 1:
            observation[2] = 1  # Danger on the up
        if heman_y < len(self.maze) - 1 and self.maze[heman_y + 1][heman_x] == 1:
            observation[3] = 1  # Danger on the down

        # Indicate the current direction of Heman
        if heman_direction == 0:  # Left
            observation[4] = 1
        elif heman_direction == 1:  # Right
            observation[5] = 1
        elif heman_direction == 2:  # Up
            observation[6] = 1
        elif heman_direction == 3:  # Down
            observation[7] = 1

        # Check for enemies in the left, right, up, and down directions
        for enemy in self.enemies:
            if not enemy['caught']:
                enemy_x, enemy_y = enemy['position']
                enemy_x //= BLOCK_SIZE
                enemy_y //= BLOCK_SIZE
                if enemy_x < heman_x:
                    observation[8] = 1  # Enemy on the left
                elif enemy_x > heman_x:
                    observation[9] = 1  # Enemy on the right
                if enemy_y < heman_y:
                    observation[10] = 1  # Enemy on the up
                elif enemy_y > heman_y:
                    observation[11] = 1  # Enemy on the down

        return observation

    def render(self):
        return self.get_observation()

    def step(self, hero_action, heman_action):
        self.current_step += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_hero_position = self.hero['position']
        old_heman_position = self.heman['position']
        self.move_hero(hero_action)
        self.move_heman(heman_action)
        self.move_enemies()
        caught_enemies = self.check_collisions()
        hero_caught = caught_enemies[0]
        heman_caught = caught_enemies[1]

        if hero_caught:
            self.hero_score += 1
            self.tot_hero_score += 1 
            hero_reward = 1000
            hero_reward -= self.current_step * 2
        else:
            uncaught_enemies = [enemy for enemy in self.enemies if not enemy['caught']]
            if uncaught_enemies:
                old_hero_distance = min([manhattan_distance(old_hero_position, enemy['position']) for enemy in uncaught_enemies])
                new_hero_distance = min([manhattan_distance(self.hero['position'], enemy['position']) for enemy in uncaught_enemies])
                if new_hero_distance < old_hero_distance:
                    hero_reward = 100  # Move closer to an enemy
                elif new_hero_distance > old_hero_distance:
                    hero_reward = -100  # Move away from an enemy
                else:
                    hero_reward = -10  # No change in distance
            else:
                hero_reward = 0  # No uncaught enemies left

        if heman_caught:
            self.heman_score += 1
            self.tot_heman_score += 1 
            heman_reward = 1000
            heman_reward -= self.current_step * 2
        else:
            uncaught_enemies = [enemy for enemy in self.enemies if not enemy['caught']]
            if uncaught_enemies:
                old_heman_distance = min([manhattan_distance(old_heman_position, enemy['position']) for enemy in uncaught_enemies])
                new_heman_distance = min([manhattan_distance(self.heman['position'], enemy['position']) for enemy in uncaught_enemies])
                if new_heman_distance < old_heman_distance:
                    heman_reward = 100  # Move closer to an enemy
                elif new_heman_distance > old_heman_distance:
                    heman_reward = -100  # Move away from an enemy
                else:
                    heman_reward = -10  # No change in distance
            else:
                heman_reward = 0  # No uncaught enemies left

        terminated = False
        if all(self.enemies_caught):
            terminated = True
            if hero_caught > heman_caught:
                hero_reward += 500
                heman_reward -= 500
            elif hero_caught < heman_caught:
                hero_reward -= 500
                heman_reward += 500

         # If not all enemies are caught after max_steps, the episode ends
        elif self.current_step >= self.max_steps:
            terminated = True
            remaining_enemies = sum(1 for caught in self.enemies_caught if not caught)
            if remaining_enemies == 0:
                # If all enemies are caught, reward the agent who caught more enemies
                if hero_caught > heman_caught:
                    hero_reward += 500
                    heman_reward -= 500
                elif hero_caught < heman_caught:
                    hero_reward -= 500
                    heman_reward += 500
            else:
                # If some enemies remain, penalize both agents
                hero_reward -= 500
                heman_reward -= 500
        truncated = False

        wall_collision_hero = self.check_wall_collision(self.hero['position'])
        wall_collision_heman = self.check_wall_collision(self.heman['position'])
        if wall_collision_hero:
            self.hero_health -= 1
            if self.hero_health == 0:
                hero_reward = -1000
                terminated = True
                truncated = False

        if wall_collision_heman:
            self.heman_health -= 1
            if self.heman_health == 0:
                heman_reward = -1000
                terminated = True
                truncated = False

        next_state = self.get_observation()

        if self.hero_health <= 0 or self.heman_health <= 0:
            terminated = True

        self.view()
        self.clock.tick(self.speed)

        return next_state, hero_reward, heman_reward, terminated, truncated, {}

    def check_wall_collision(self, position):
        x, y = position
        x //= BLOCK_SIZE
        y //= BLOCK_SIZE
        if self.maze[y][x] == 1:
            return True
        return False

    def move_hero(self, action):
        x = self.hero['position'].x
        y = self.hero['position'].y
        x_next = x
        y_next = y

        # if action == 0:
        #     x_next = x + BLOCK_SIZE
        # elif action == 1:
        #     x_next = x - BLOCK_SIZE
        # elif action == 2:
        #     y_next = y + BLOCK_SIZE
        # elif action == 3:
        #     y_next = y - BLOCK_SIZE

        if action == 0:
            x_next = x + BLOCK_SIZE
            self.hero['direction'] = 1  # Right
        elif action == 1:
            x_next = x - BLOCK_SIZE
            self.hero['direction'] = 0  # Left
        elif action == 2:
            y_next = y + BLOCK_SIZE
            self.hero['direction'] = 2  # Down
        elif action == 3:
            y_next = y - BLOCK_SIZE
            self.hero['direction'] = 3  # Up

        if 0 <= x_next < self.w and 0 <= y_next < self.h:
            if self.maze[y_next // BLOCK_SIZE][x_next // BLOCK_SIZE] == 1:
                self.hero_health -= 0.5
                if self.hero_health <= 0:
                    return False
            else:
                self.hero['position'] = Point(x_next, y_next)
        else:
            self.hero['position'] = Point(x, y)

        return True

    def move_heman(self, action):
        x = self.heman['position'].x
        y = self.heman['position'].y
        x_next = x
        y_next = y

        # if action == 0:
        #     x_next = x + BLOCK_SIZE
        # elif action == 1:
        #     x_next = x - BLOCK_SIZE
        # elif action == 2:
        #     y_next = y + BLOCK_SIZE
        # elif action == 3:
        #     y_next = y - BLOCK_SIZE

        if action == 0:
            x_next = x + BLOCK_SIZE
            self.heman['direction'] = 1  # Right
        elif action == 1:
            x_next = x - BLOCK_SIZE
            self.heman['direction'] = 0  # Left
        elif action == 2:
            y_next = y + BLOCK_SIZE
            self.heman['direction'] = 2  # Down
        elif action == 3:
            y_next = y - BLOCK_SIZE
            self.heman['direction'] = 3  # Up

        if 0 <= x_next < self.w and 0 <= y_next < self.h:
            if self.maze[y_next // BLOCK_SIZE][x_next // BLOCK_SIZE] == 1:
                self.heman_health -= 0.5
                if self.heman_health <= 0:
                    return False
            else:
                self.heman['position'] = Point(x_next, y_next)
        else:
            self.heman['position'] = Point(x, y)

        return True

    def move_enemies(self):
        if self.enemy_move_counter % 2 == 0:
            for enemy in self.enemies:
                if not enemy['caught']:
                    x = enemy['position'].x
                    y = enemy['position'].y
                    action = random.randint(0, 3)
                    x_next = x
                    y_next = y

                    if action == 0:
                        x_next = x + BLOCK_SIZE
                    elif action == 1:
                        x_next = x - BLOCK_SIZE
                    elif action == 2:
                        y_next = y + BLOCK_SIZE
                    elif action == 3:
                        y_next = y - BLOCK_SIZE

                    if 0 <= x_next < self.w and 0 <= y_next < self.h:
                        if self.maze[y_next // BLOCK_SIZE][x_next // BLOCK_SIZE] == 0:
                            enemy['position'] = Point(x_next, y_next)
        self.enemy_move_counter += 1 

    def check_collisions(self):
        self.hero_caught_enemies = []
        self.heman_caught_enemies = []
        for i, enemy in enumerate(self.enemies):
            if not enemy['caught']:
                if enemy['position'] == self.hero['position']:
                    self.hero_caught_enemies.append(i)
                if enemy['position'] == self.heman['position']:
                    self.heman_caught_enemies.append(i)

        for i in self.hero_caught_enemies:
            self.enemies_caught[i] = True
            self.enemies[i]['caught'] = True  # Update the caught status

        for i in self.heman_caught_enemies:
            self.enemies_caught[i] = True
            self.enemies[i]['caught'] = True  # Update the caught status

        hero_caught = bool(self.hero_caught_enemies)
        heman_caught = bool(self.heman_caught_enemies)

        for potion in self.health_potions:
            if potion['position'] == self.hero['position']:
                if self.hero_health < 100:
                    self.hero_health += 4
                    if self.hero_health > 100:
                        self.hero_health = 100
                potion['found'] = True
                self.health_potions.remove(potion)
                break
            if potion['position'] == self.heman['position']:
                if self.heman_health < 100:
                    self.heman_health += 4
                    if self.heman_health > 100:
                        self.heman_health = 100
                potion['found'] = True
                self.health_potions.remove(potion)
                break

        return [hero_caught, heman_caught]
    
    def view(self):
        self.display.blit(self.background, (0, 0))
        self.draw_maze()
        self.display.blit(self.hero_img, (self.hero['position'].x, self.hero['position'].y))
        self.display.blit(self.heman_img, (self.heman['position'].x, self.heman['position'].y))
        for enemy in self.enemies:
            if not enemy['caught']:
                self.display.blit(self.enemy_img, (enemy['position'].x, enemy['position'].y))
        for potion in self.health_potions:
            if not potion['found']:
                self.display.blit(self.hp_img, (potion['position'].x, potion['position'].y))
        text_1 = font.render(f"Score: {self.rew}", True, WHITE)
        text_2 = font.render(f"Steps: {self.current_step}", True, WHITE)
        text_3 = font.render(f"Total Hero: {self.tot_hero_score}", True, WHITE)
        text_4 = font.render(f"Total Heman: {self.tot_heman_score}", True, WHITE)
        text_5 = font.render(f"Hero: {self.hero_score}", True, WHITE)
        text_6 = font.render(f"Heman: {self.heman_score}", True, WHITE)
        text_7 = font.render(f"Enemies Left: {sum(1 for caught in self.enemies_caught if not caught)}", True, WHITE)
        text_8 = font.render(f"Episode: {episode+1}", True, WHITE)

        # self.display.blit(text_1, [0, 0])
        self.display.blit(text_2, [0, 0])
        self.display.blit(text_3, [150, 0])
        self.display.blit(text_4, [310, 0])
        self.display.blit(text_5, [510, 0])
        self.display.blit(text_6, [610, 0])
        self.display.blit(text_8, [730, 0])
        pygame.display.flip()
    def draw_maze(self):
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                if self.maze[y][x] == 1:
                    self.display.blit(self.wall_img, (x * BLOCK_SIZE, y * BLOCK_SIZE))

def close(self):
    pygame.quit()


# Create the environment
env = ShaktimanEnv(speed=1000000, num_health_potions=5)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of episodes to test
num_episodes = 10
cummulative_hero_reward=[]
cummulative_heman_reward=[]
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    terminated = False
    truncated = False
    episode_hero_reward = 0
    episode_heman_reward = 0
    

    while not terminated:
        # Select actions using the trained models
        with torch.no_grad():
            hero_q_values = Q_net(state.unsqueeze(0))
            hero_action = torch.argmax(hero_q_values, dim=1).item()

            # hero_action = env.action_space.sample()

            
            state = torch.tensor(state, dtype=torch.float32, device=device)
            policy = actor_net(state)
            action_prob = policy
            heman_action = action_prob.multinomial(num_samples=1).item()

            # heman_action = env.action_space.sample()

        next_state, hero_reward, heman_reward, terminated, truncated, _ = env.step(hero_action, heman_action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        state = next_state
        episode_hero_reward += hero_reward
        episode_heman_reward += heman_reward
    cummulative_hero_reward.append(episode_hero_reward)
    cummulative_heman_reward.append(episode_heman_reward)
    # print(cummulative_hero_reward)
    # print(cummulative_heman_reward)

    
    
    
    print(f"Episode {episode + 1}: Hero Reward = {episode_hero_reward}, Heman Reward = {episode_heman_reward}")
    if episode_hero_reward > episode_heman_reward:
        print("Hero Won !!",'\n')
    else:
        print("Heman Won !!",'\n')
# print(cummulative_hero_reward)
# print(cummulative_heman_reward)
plt.plot(cummulative_hero_reward, label='Hero')
plt.plot(cummulative_heman_reward, label='Heman')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Rewards Comparison')
plt.legend()
plt.savefig('./Plots/cummulative_hero_testing.png')
plt.savefig('./Plots/cummulative_heman_testing.png')
plt.show()


env.close()