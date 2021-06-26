# Project 3: Collaboration and competition

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

Check out the video of the trained agent in action: 
[![Trained Agents](https://img.youtube.com/vi/Tk6SflL-eKI/0.jpg)](https://www.youtube.com/watch?v=Tk6SflL-eKI)

## Observation space

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  

## Action space

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

## Rewards

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes, however this notebook attempts to achieve a score of +15

# Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Extract the zip folder and place the executable in the `Tennis_Windows_x86_64` folder with the name of `Tennis.exe`.

3. Open the terminal and navigate to the `p3_collab-compet`

4. Execute `pip install -r requirements.txt`, this command will install all the dependencies

5. Start the Jupyter notebook by running the command `jupyter notebook`, a new browser tab/window will open with the running notebook

6. Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

