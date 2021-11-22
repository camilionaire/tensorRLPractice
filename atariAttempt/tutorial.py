
from ctypes import windll
import gym
import random
import ale_py # not 100 if need this.
import numpy as np

# for building and compiling(Adam) the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

# for building the agent(I think?..)
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def example(env, actions):
	episodes = 5
	for episode in range(1, episodes+1):
		state = env.reset()
		done = False
		score = 0

		while not done:
			env.render()
			# better way, I think. more... dynamic.
			# action = random.choice([0, 1, 2, 3, 4, 5])
			action = random.randint(0, actions-1)
			n_state, reward, done, info = env.step(action)
			score+=reward

		# updated with that new way of formatting text I learned.
		print(f'Episode: {episode} Score: {score}')

	env.close()

def build_model(height, width, channels, actions):

	model = Sequential()

	# num of filters, size, strides 4,4 means diagonal, activation, input shape (3 allows mult images), hwc
	model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3, height, width, channels)))
	model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
	model.add(Convolution2D(64, (3,3), activation='relu'))
	model.add(Flatten())

	# dense layer aka fully connected layer
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(actions, activation='linear'))

	return model

def build_agent(model, actions):
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
	memory = SequentialMemory(limit=1000, window_length=3)
	dqn = DQNAgent(model=model, memory=memory, policy=policy,
				enable_dueling_network=True, dueling_type='avg', 
					nb_actions=actions, nb_steps_warmup=1000)
	return dqn

def main():

	print("\n\nIgnore any warnings about tensorflow above, unless you have nVidia")
	print("BEGINNING OF PROGRAMM!!!\n")
	print('gym:', gym.__version__)
	print('ale_py:', ale_py.__version__)

	# v0 is just the image data(? maybe?...)
	env = gym.make('SpaceInvaders-v0')
	height, width, channels = env.observation_space.shape
	actions = env.action_space.n

	print(f"height: {height} width: {width} channels: {channels}")
	print(f"actions: {actions}")

	print(env.unwrapped.get_action_meanings())
	# example random actions from function above.
	# example(env, actions)

	model = build_model(height, width, channels, actions)

	model.summary()

	# build the agent for the model / actions and compile and then fit.
	dqn = build_agent(model, actions)
	dqn.compile(Adam(learning_rate=1e-4))
	dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

	scores = dqn.test(env, nb_episodes=10, visualize=True)
	print(np.mean(scores.history['episode_reward']))

	dqn.save_weights('SavedWeights/10k-Fast/dqn_weights.h5f')

	del model, dqn

	dqn.load_weights('SavedWeights/10k-Fast/dqn_weights.h5f')

	env.close()

if __name__ == "__main__":
	main()