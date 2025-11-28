import numpy as np
import subprocess
import pickle
import os
import gymnasium as gym
from gymnasium import spaces



def safe_pickle_load(filename):
    """Safely load pickle file handling corruption and race conditions"""
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        return None

        
class SC2ENV(gym.Env):
    def __init__(self):
        super(SC2ENV, self).__init__()
        self.MAX_ITER = 1000
        self.MAP_SHAPE = (176, 184, 3)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.MAP_SHAPE, dtype=np.uint8)
        
    def _send_move(self, move):
        safe_word = 0
        while True and safe_word < self.MAX_ITER:
            try:
                sarsa = safe_pickle_load('sarsa.pkl')
                if sarsa and sarsa['moves'] is None:
                    sarsa['moves'] = move
                    with open('sarsa.pkl', 'wb') as f:
                        pickle.dump(sarsa, f)
                    break
                safe_word += 1
            except Exception as e:
                print(f'Error in the environment, _send_move method - {e}')

    def _get_state(self):
        safe_word = 0
        while True and safe_word < self.MAX_ITER:
            try:
                sarsa = safe_pickle_load('sarsa.pkl')
                if sarsa['moves'] is not None:
                    observation = sarsa['game_map']
                    reward = sarsa['reward']
                    terminated = sarsa['game_finished']
                    break
                safe_word += 1

            except Exception as e:
                print(f'Error in the environment, _get_state method - {e}')
                # If some error occurs, we create a default state
                self._dump_default_state()
        if observation is None:
            observation = np.zeros(self.MAP_SHAPE, dtype=np.uint8)
        if reward is None:
            reward = 0
        if terminated is None:
            terminated = False
        return observation, reward, terminated

    def _dump_default_state(self):
        observation = np.zeros(self.MAP_SHAPE, dtype=np.uint8)
        data = {
            "game_map": observation,
            "reward": 0,
            "moves": 5,
            "game_finished": False
            }
        
        with open('sarsa.pkl', 'wb') as f:
            pickle.dump(data, f)

    def step(self, move):
        self._send_move(move)
        observation, reward, terminated = self._get_state()
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print('-' * 20)
        print("RESETTING ENVIRONMENT")
        print('-' * 20)
        observation = np.zeros(self.MAP_SHAPE, dtype=np.uint8)
        game_state_dict = {
            "game_map": observation,
            "reward": 0,
            "moves": 5,
            "game_finished": False
            } 

        with open('sarsa.pkl', 'wb') as f:
            pickle.dump(game_state_dict, f)

        subprocess.Popen(['python', 'ReinforceBot.py'])
        info = {}
        return observation, info
