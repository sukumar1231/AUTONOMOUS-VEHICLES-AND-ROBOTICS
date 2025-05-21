import pygame
import numpy as np
import random

# Init Pygame
clock = pygame.time.Clock()
pygame.init()
WIDTH, HEIGHT = 600, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AV Car Simulation")

# Grid world settings
GRID_SIZE = 50
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Q-learning settings
actions = ['left', 'right', 'straight']
q_table = {}
learning_rate = 0.1
discount = 0.9
epsilon = 0.1

# Environment setup
lanes = [10, 10]  # simplified for this grid
obstacles = [(random.randint(0, COLS-1), random.randint(5, ROWS-2)) for _ in range(15)]
car_pos = [20, 30]

# === Drawing Functions ===
def draw_grid():
    win.fill(GRAY)
    for x in range(COLS):
        for y in range(ROWS):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(win, WHITE, rect, 1)
    for obs in obstacles:
        pygame.draw.rect(win, RED, (obs[0] * GRID_SIZE + 5, obs[1] * GRID_SIZE + 5, 40, 40))

def draw_car(x, y):
    pygame.draw.rect(win, GREEN, (x * GRID_SIZE + 5, y * GRID_SIZE + 5, 40, 40))

# === Logic Functions ===
def reward_fn(pos):
    car_rect = pygame.Rect(pos[0] * GRID_SIZE + 5, pos[1] * GRID_SIZE + 5, 40, 40)
    for obs in obstacles:
        obs_rect = pygame.Rect(obs[0] * GRID_SIZE + 5, obs[1] * GRID_SIZE + 5, 40, 40)
        if car_rect.colliderect(obs_rect):
            return -10
    if pos[0] not in lanes:
        return -5
    return 1

def get_state(pos):
    left = (pos[0]-1, pos[1]+1) in obstacles or (pos[0]-1, pos[1]+2) in obstacles
    front = (pos[0], pos[1]+1) in obstacles or (pos[0], pos[1]+2) in obstacles
    right = (pos[0]+1, pos[1]+1) in obstacles or (pos[0]+1, pos[1]+2) in obstacles
    return (pos[0], pos[1], left, front, right)

def get_max_q(state):
    if state not in q_table:
        q_table[state] = {a: 0 for a in actions}
    return max(q_table[state], key=q_table[state].get)

def choose_action(state):
    if state not in q_table:
        q_table[state] = {a: 0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)
    return get_max_q(state)

def move(pos, action):
    x, y = pos
    if action == 'left':
        x = max(0, x - 1)
    elif action == 'right':
        x = min(COLS - 1, x + 1)
    return [x, y - 1]

# === Main loop ===
run = True
episodes = 0
max_episodes = 1000

while run and episodes < max_episodes:
    car_pos = [10, 29]
    total_reward = 0

    for step in range(100):
        pygame.time.delay(9)
        clock.tick(6)

        draw_grid()
        draw_car(car_pos[0], car_pos[1])
        pygame.display.update()

        state = get_state(car_pos)
        action = choose_action(state)
        new_pos = move(car_pos, action)
        r = reward_fn(new_pos)

        if r == -10:
            pygame.time.delay(300)
            car_pos = [car_pos[0], min(car_pos[1] + 1, ROWS - 1)]
            alt_action = random.choice(actions)
            new_pos = move(car_pos, alt_action)
            r = reward_fn(new_pos)
            new_state = get_state(new_pos)
            if new_state not in q_table:
                q_table[new_state] = {a: 0 for a in actions}
            old_q = q_table[state][action]
            next_max = max(q_table[new_state].values())
            q_table[state][action] = old_q + learning_rate * (r + discount * next_max - old_q)
            car_pos = new_pos
            total_reward += r
        else:
            new_state = get_state(new_pos)
            if new_state not in q_table:
                q_table[new_state] = {a: 0 for a in actions}
            old_q = q_table[state][action]
            next_max = max(q_table[new_state].values())
            q_table[state][action] = old_q + learning_rate * (r + discount * next_max - old_q)
            car_pos = new_pos
            total_reward += r

        if car_pos[1] <= 0 or r == -10:
            break

    episodes += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()
print("Training complete.")
