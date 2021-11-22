import gym
import random

# numpy arrays
import numpy as np
# import tensorflow as tf
# import the sequential model with keras
from tensorflow.keras.models import Sequential
# two diff types of layers, dense and flatten node
from tensorflow.keras.layers import Dense, Flatten
# optimizer that we use to train our drl model
from tensorflow.keras.optimizers import Adam

# RL DEPENDENCIES
# going to be using policy based rl, specifically the Boltzmann Policy
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
# for dqn agent need to maintain some memory
from rl.memory import SequentialMemory

# builds a model to take in number of states and return number of actions.
def build_model(states, actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	return model

# pass in model defined above, and number of actions.
def build_agent(model, actions):
	# set up policy, memory and dqn agent
	policy = BoltzmannQPolicy()
	# NOTE dropped down limit from 50K in tutorial to 10K.
	memory = SequentialMemory(limit=10000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy,
			nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
	return dqn



def main():
	env = gym.make('CartPole-v0')
	states = env.observation_space.shape[0]
	actions = env.action_space.n

	print("states:",states)
	print("actions:",actions)

	episodes = 10
	for episode in range(1, episodes):
		state = env.reset()
		done = False
		score = 0

		while not done:
			env.render()
			action = random.choice([0, 1])
			n_state, reward, done, infor = env.step(action)
			score+=reward
		print(f'Episode: {episode}, Score: {score}')

	env.close()

	model = build_model(states, actions)

	model.summary()

	# build agent function from above
	dqn = build_agent(model, actions)
	# lr is deprecated, use learning_rate instead.
	# comile with adam... mean absolute error
	dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
	# keep up the training, environment, steps, no visualize... little bit of log info
	# dropped down from the tutorial of 50K to 10K
	dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)

	scores = dqn.test(env, nb_episodes=100, visualize=False)
	print(np.mean(scores.history['episode_reward']))

	# if you want to see if visualized a few times...

if __name__ == "__main__":
	main()