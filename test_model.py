from stable_baselines3 import PPO
from SCenv import SC2ENV


def main():
    LOAD_MODEL = "S:/ppomodele/PRACA1736191570/160000"

    if LOAD_MODEL == "BRAK_SCIEZKI":
        print("Brak sciezki")
    env = SC2ENV()
    model = PPO.load(LOAD_MODEL)
    obserwacja = env.reset()
    rozgrywka_skonczona = False
    while not rozgrywka_skonczona:
        ruchy, _mapy_gry = model.predict(obserwacja)
        obserwacja, rewards, rozgrywka_skonczona, info = env.step(ruchy)
    return 

while True:
    main()
