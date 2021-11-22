
import gym
import random
import ale_py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def example(env, actions):
	episodes = 5
	for episode in episodes:
		state = env.reset()
		done = False
		score = 0

		while not done:
			env.render()
			# better way, I think. more... dynamic.
			# action = random.choice([0, 1, 2, 3, 4, 5])
			action = random.randint(actions)
			n_state, reward, done, info = env.step(action)
			score+=reward

		# updated with that new way of formatting text I learned.
		print(f'Episode: {episode} Score: {score}')

	env.close()


def main():

	print("\n\nIgnore any warnings about tensorflow above, unless you have nVidia")
	print("BEGINNING OF PROGRAMM!!!\n")
	print('gym:', gym.__version__)
	print('ale_py:', ale_py.__version__)

	env = gym.make('SpaceInvaders-v0')
	height, width, channels = env.observation_space.shape
	actions = env.action_space.n

	print(f"height: {height} width: {width} channels: {channels}")
	print(f"actions: {actions}")

	print(env.unwrapped.get_action_meanings())


if __name__ == "__main__":
	main()