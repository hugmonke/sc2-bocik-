import numpy as np
import subprocess
import pickle
import os
import gym
from gym import spaces

class SC2ENV(gym.Env):
	def __init__(self):
		super(SC2ENV, self).__init__()
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

	def step(self, ruch):
		czeka_na_ruchy = True
		while czeka_na_ruchy:
			try:
				with open('sarsa.pkl', 'rb') as f:
					sarsa = pickle.load(f)
					
					if sarsa['ruchy'] is not None:
						czeka_na_ruchy = True
					else:
						czeka_na_ruchy = False
						sarsa['ruchy'] = ruch
						with open('sarsa.pkl', 'wb') as f:
							pickle.dump(sarsa, f)
			except Exception as e:
				pass

		czeka_na_mape = True
		while czeka_na_mape:
			try:
				if os.path.getsize('sarsa.pkl') > 0:
					with open('sarsa.pkl', 'rb') as f:
						sarsa = pickle.load(f)
						if sarsa['ruchy'] is None:
							czeka_na_mape = True
						else:
							stan = sarsa['mapa_gry']
							nagroda = sarsa['nagroda']
							rozgrywka_skonczona = sarsa['rozgrywka_skonczona']	
							czeka_na_mape = False

			except Exception as e:
				czeka_na_mape = True   
				mapa_gry = np.zeros((128, 128, 3), dtype=np.uint8)
				obserwacja = mapa_gry
				data = {"mapa_gry": mapa_gry, 
			            "nagroda": 0, 
						"ruchy": 5, 
						"rozgrywka_skonczona": False}  
				
				with open('sarsa.pkl', 'wb') as f:
					pickle.dump(data, f)

				stan = mapa_gry
				nagroda = 0
				rozgrywka_skonczona = False
				#ruch = 5

		info = {}
		obserwacja = stan
		return obserwacja, nagroda, rozgrywka_skonczona, info


	def reset(self):
		#super().reset(seed=seed)
		print("----RESETOWANIE SRODOWISKA----")
		pusta_mapa = np.zeros((128, 128, 3), dtype=np.uint8)
		obserwacja = pusta_mapa
		dane_rozgrywki = {"mapa_gry": pusta_mapa, 
					"nagroda": 0, 
					"ruchy": 5, 
					"rozgrywka_skonczona": False} 
		
		with open('sarsa.pkl', 'wb') as f:
			pickle.dump(dane_rozgrywki, f)

		subprocess.Popen(['python', 'ReinforceBot.py'])
		return obserwacja