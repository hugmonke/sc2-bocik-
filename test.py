from stable_baselines3 import PPO
from SC2Env import SC2ENV


def main():
    LOAD_MODEL = "PRACA1736191570/160000"
    env = SC2ENV()
    model = PPO.load(LOAD_MODEL)
    observation = env.reset()
    terminated = False
    while not terminated:
        moves, _mapy_gry = model.predict(observation)
        observation, rewards, terminated, info = env.step(moves)
    return None

while True:
    main()
