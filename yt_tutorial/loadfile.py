import gym
import rltutorial as rlt

# optimizer that we use to train our drl model
from tensorflow.keras.optimizers import Adam


def main():
	env = gym.make('CartPole-v0')
	actions = env.action_space.n
	states = env.observation_space.shape[0]
	model = rlt.build_model(states, actions)
	dqn = rlt.build_agent(model, actions)
	dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

	dqn.load_weights('dqn_weights.h5f')

	dqn.test(env, nb_episodes=5, visualize=True)

	env.close()

if __name__ == "__main__":
	main()