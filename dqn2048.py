import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
from game2048 import Game2048Env

# Using cuda to accelerate training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class DQN(nn.Module):
    ### Deep Q-Learning Network
    def __init__(self):
        super(DQN, self).__init__()

        self.conv_a = nn.Conv2d(16, 128, kernel_size=(1, 2))        # Convolutional layers
        self.conv_b = nn.Conv2d(16, 128, kernel_size=(2, 1))

        self.conv_aa = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_ab = nn.Conv2d(128, 128, kernel_size=(2, 1))

        self.conv_ba = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_bb = nn.Conv2d(128, 128, kernel_size=(2, 1))

        self.fc = nn.Sequential(         # Linear and Relu
            nn.Linear(7424, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
      # Forward function of the layer
        x_a = F.relu(self.conv_a(x))
        x_b = F.relu(self.conv_b(x))

        x_aa = F.relu(self.conv_aa(x_a))
        x_ab = F.relu(self.conv_ab(x_a))

        x_ba = F.relu(self.conv_ba(x_b))
        x_bb = F.relu(self.conv_bb(x_b))

        sh_a = x_a.shape
        sh_aa = x_aa.shape
        sh_ab = x_ab.shape
        sh_b = x_b.shape
        sh_ba = x_ba.shape
        sh_bb = x_bb.shape

        x_a = x_a.view(sh_a[0], sh_a[1]*sh_a[2]*sh_a[3])
        x_aa = x_aa.view(sh_aa[0], sh_aa[1]*sh_aa[2]*sh_aa[3])
        x_ab = x_ab.view(sh_ab[0], sh_ab[1]*sh_ab[2]*sh_ab[3])
        x_b = x_b.view(sh_b[0], sh_b[1]*sh_b[2]*sh_b[3])
        x_ba = x_ba.view(sh_ba[0], sh_ba[1]*sh_ba[2]*sh_ba[3])
        x_bb = x_bb.view(sh_bb[0], sh_bb[1]*sh_bb[2]*sh_bb[3])

        concat = torch.cat((x_a, x_b, x_aa, x_ab, x_ba, x_bb), dim=1)

        output = self.fc(concat)

        return output


def change_values(X):
    # Transform input of the DQN (normalization)
    power_mat = np.zeros(shape=(1, 16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j] == 0):
                power_mat[0][0][i][j] = 1.0
            else:
                power = int(math.log(X[i][j], 2))
                power_mat[0][power][i][j] = 1.0
    return power_mat

class Memory(object):
  ## Class for replay buffer
    def __init__(self, memory_size, array):
        self.memory_size = memory_size
        self.buffer = collections.deque(array, maxlen=self.memory_size)

    def add(self, experience):
      # Add to buffer
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
      ## Sample min(batch_size, len(buffer)) elements for buffer
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


def training(n_epochs, reward_mode, online_dic, target_dic, epsilon, memory_buffer, opti, path="/content/gdrive/My Drive/2048/", cont=True, print_rate=100):
    # Training the agent (we input parameters coming from previous training)

    GAMMA = 0.99
    EXPLORE = 10000
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0001
    REPLAY_MEMORY = 50000   # Size of replay buffer
    BATCH = 16  # Length of batch extracted from buffer

    UPDATE_STEPS = 4

    begin_learn = False
    learn_steps = 0
    episode_reward = 0
    scores = []
    max_tiles = []

    ENV_NAME = '2048'
    env = Game2048Env()
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    if cont:
      ## In this case, we load previous training parameters to continue the training
        epsilon = np.float(np.load(path+epsilon))
        memory_replay = Memory(REPLAY_MEMORY, collections.deque(
            np.load(path+memory_buffer, allow_pickle=True)))

        onlineQNetwork = DQN().to(device)
        targetQNetwork = DQN().to(device)
        onlineQNetwork.load_state_dict(torch.load(path+online_dic))
        targetQNetwork.load_state_dict(torch.load(path+target_dic))

        optimizer = torch.load(path + opti)

    else:
      ## Start of the training

        epsilon = INITIAL_EPSILON
        memory_replay = Memory(REPLAY_MEMORY, np.array([]))
        onlineQNetwork = DQN().to(device)
        targetQNetwork = DQN().to(device)
        targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

        optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    for epoch in range(n_epochs):


        state = env.reset()
        episode_reward = 0
        done = False

        while not done:

            x = change_values(state)
            x = torch.from_numpy(np.flip(x, axis=0).copy()).to(device)

            # Epsilon-greedy approach for the policy
            if random.random() < epsilon:
                action = random.randint(0, 3)
                next_state, reward, done, _ = env.step(action)
                #print(next_state)
                
                while (state != next_state).all():
                    action = random.randint(0, 3)
                    next_state, reward, done, _ = env.step(action)
                    #
                    #print(done)
            else:
                output = onlineQNetwork.forward(x)
                for action in output.argsort()[0].cpu().numpy()[::-1]:
                    next_state, reward, done, _ = env.step(action)
                    if (state == next_state).all():
                        break

            episode_reward += reward
            memory_replay.add((change_values(state), change_values(next_state), action, reward, done))  # Adding data to the replay buffer

            if memory_replay.size() > 128:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                batch = memory_replay.sample(BATCH)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.FloatTensor(batch_state).squeeze(1).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).squeeze(1).to(device)
                batch_action = torch.Tensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.Tensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    targetQ_next = targetQNetwork(batch_next_state)
                    y = batch_reward + (1 - batch_done) * GAMMA * torch.max(targetQ_next, dim=1, keepdim=True)[0]      # Q-learning update

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            state = next_state

        scores.append(episode_reward)
        max_tiles.append(np.max(state))

        env.render()
        print("Game "+str(epoch)+", Episode reward: "+str(episode_reward))
        if epoch % print_rate == 0:
            env.render()
            #save_data(onlineQNetwork, targetQNetwork, optimizer, scores, max_tiles, epsilon, memory_replay, 0, 0, final = False)  #Uncomment to save data (not useful if you punctually train the agent)
            print("Game "+str(epoch)+", Episode reward: "+str(episode_reward))

    return(onlineQNetwork, targetQNetwork, optimizer, scores, max_tiles, epsilon, memory_replay)


def save_data(onlineQNetwork, targetQNetwork, optimizer, scores, max_tiles, epsilon, memory_replay, reward, run, path="/2022_AI_project", final=True):
      ## Saves data in drive (previously mounted)
  if final:
    suffix = '_reward'+str(reward)+'_run' + str(run)
  else: 
    suffix = ''

  torch.save(onlineQNetwork.state_dict(), path + "online"+suffix)
  torch.save(targetQNetwork.state_dict(), path+ 'target' + suffix)
  torch.save(optimizer,path+'opti'+ suffix)
  np.save(path+'scores' + suffix, scores)
  np.save(path+'max_tiles' + suffix, max_tiles)
  np.save(path+'eps' + suffix, epsilon)
  np.save(path+'mem' + suffix, np.array(memory_replay.buffer))
  return()

  
onlineQNetwork, targetQNetwork, optimizer, scores, max_tiles, epsilon, memory_replay = training(1000, 'nb_merge_max_tile', 'online', 'target', 'eps.npy', 'mem.npy', 'opti', cont=False, print_rate=100)
save_data(onlineQNetwork, targetQNetwork, optimizer,scores, max_tiles, epsilon, memory_replay, 3, 1)


def moving_average(a, wind=25):
    # Returns the moving average array associated to a with a window wind
    ret = np.cumsum(a, dtype=float)
    ret[wind:] = ret[wind:] - ret[:-wind]
    return ret[wind - 1:] / wind


#print the curve of max tile for the three reward
path = "/2022_AI_project"


suffix = "_reward3_run1.npy"
scores_3 = np.load(path + "scores" + suffix)
max_tiles_3 = np.load(path+"max_tiles" + suffix)

plt.figure(0)
plt.xlabel('Number of games')
plt.ylabel('Moving average of max_tiles (window size 25)')
plt.plot(moving_average(max_tiles_3), label='Reward 2')
plt.legend(loc='best')
plt.title('Max tiles for the three rewards (moving average)')


def play_game(env, QNetwork, render=True):
    ## Plays one unique game given an environment and a trained DQN
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:

        x = change_values(state)
        x = torch.from_numpy(np.flip(x, axis=0).copy()).to(device)
        output = QNetwork.forward(x)
        for action in output.argsort()[0].cpu().numpy()[::-1]:
            next_state, reward, done, _ = env.step(action)
            if (state == next_state).all() == False:
                break

        episode_reward += reward
        state = next_state

    if render:
      env.render()
      print("Score: "+str(episode_reward))

    return(state)


def results(reward_mode):
    # Displays results (proportions of games reaching each existing level of max_tile) given a reward_mode
    if reward_mode == 'nb_merge_max_tile':
      reward_num = 3
      run = 1
    elif reward_mode == 'score':
      reward_num = 1
      run = 2
    elif reward_mode == 'nb_empty_tiles':
      reward_num = 4
      run = 3

    test_env = Game2048Env()
    path = "/content/gdrive/My Drive/2048/"
    QNetwork = DQN().to(device)
    filename = 'online_reward' + str(reward_num) + '_run' + str(run)
    QNetwork.load_state_dict(torch.load(path + filename))

    dic_max = {}
    for i in range(13):
      dic_max[2**i] = 0

    n_games = 1000

    for k in range(n_games):
      if (k % 50 == 0):
        print(str(k) + " games played ")
      grid = play_game(test_env, QNetwork, render=False)
      max_tile = np.max(grid)
      dic_max[max_tile] = dic_max[max_tile]+1

    for key in dic_max:
      dic_max[key] = dic_max[key]/n_games

    return(dic_max)


results('nb_merge_max_tile')
results('score')
results('nb_empty_tiles')
