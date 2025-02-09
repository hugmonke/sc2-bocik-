from stable_baselines3 import PPO
from SCenv import SC2ENV
import os
import time


def main():
	nazwamodelu = f"PRACA{int(time.time())}"
	sciezkamodelu = f"S:/ppomodele/{nazwamodelu}/"
	sciezkalogow = f"S:/ppologi/{nazwamodelu}/"


	if not os.path.exists(sciezkamodelu):
		os.makedirs(sciezkamodelu)

	if not os.path.exists(sciezkalogow):
		os.makedirs(sciezkalogow)

	env = SC2ENV()

	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=sciezkalogow, device='cpu')

	KROKI = 10000
	iteracja = 0
	while True:
		print("ITERACJA UCZENIA: ", iteracja)
		iteracja += 1
		print("00")
		model.learn(total_timesteps=KROKI, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{sciezkamodelu}/{KROKI*iteracja}")
		print("11")

main()