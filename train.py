import os
import time
from stable_baselines3 import PPO
from SC2Env import SC2ENV

def print_header(message):
    """Print a formatted header message."""
    print('-' * 40)
    print(message)
    print('-' * 40)
    
def main():
    """Main training loop for SC2 reinforcement learning agent."""
    timestamp = int(time.time())
    model_name = f"MODEL_{timestamp:.4f}"
    model_save_path = f"models/{model_name}/"
    log_save_path = f"logs/{model_name}/"
    TRAINING_STEPS = 10000
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    
    env = SC2ENV()
    model = PPO(policy='MlpPolicy'
                , env=env
                , verbose=1
                , tensorboard_log=log_save_path
                , device='cpu')
    
    iteration = 0
    while True:
        print_header(f"LEARNING ITERATION: {iteration}")
        try:
            model.learn(total_timesteps=TRAINING_STEPS
                        , reset_num_timesteps=False
                        , tb_log_name="PPO")
            
            checkpoint_path = f"{model_save_path}/{TRAINING_STEPS * iteration}"
            model.save(checkpoint_path)
            print(f"Model saved to: {checkpoint_path}")
            
            iteration += 1
            
        except Exception as e:
            print(f"Training error: {e}")
            break

if __name__ == "__main__":
    main()