# StarCraft II Reinforcement Learning Bot

A StarCraft II AI bot using Proximal Policy Optimization (PPO) from Stable-Baselines3 for reinforcement learning.

## Overview

This project implements a reinforcement learning agent that plays StarCraft II using the PySC2 library. The bot currently uses a combination of hardcoded strategies and learned behaviors through PPO. The long-term goal is to develop a fully autonomous agent capable of making all possible in-game decisions.

## Features

- **PPO Reinforcement Learning** using Stable-Baselines3
- **Mixed Strategy Approach**: Combines hardcoded behaviors with learned policies
- **Real-time Game Integration** through BurnySC2
- **Modular Architecture** for easy strategy development
- **Training Progress Tracking** with TensorBoard logging

## Requirements

- **Python**: Developed on 3.13.5
- **StarCraft II**: Latest version installed
- **BurnySC2**: Python StarCraft II API burnysc2=7.1.1
- **Stable-Baselines3**: Reinforcement learning algorithms stable-baselines3=2.7.0
- **Gymnasium**: Reinforcement learning environment interface gymnasium=1.2.2

## Installation

1. **Install StarCraft II**
   - Download and install StarCraft II from [Blizzard](https://starcraft2.com/)
   - Ensure the game is fully updated

2. **Download Game Maps**
   ```bash
   # Clone the StarCraft II maps repository
   git clone https://github.com/Blizzard/s2client-proto.git


