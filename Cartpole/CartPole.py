import numpy as np
import gym
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
# from PIL import Image
# import tensorflow as tf


class DQL():

    def __init__(self, env):
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.model()

    def model(self):
        input = env.observation_space.shape[0]
        output = env.action_space.n
        model = Sequential()
        model.add(Dense(128, input_dim= input, activation='tanh'))
        model.add(Dense(output, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQL(env)

    batch_size = 32
    episodes = 10

    for e in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, 4])
        time = 0

        while True:
            action = agent.action(state)
            # step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # storage
            agent.remember(state, action, reward, next_state, done)
            # update state
            state = next_state
            # training with experience replay
            agent.replay(batch_size)
            # adjust epsilon
            agent.adjust_epsilon()

            time += 1

            if done:
                print('episode: {}, time: {}'.format(e, time))
                break

# Test

trained_model = agent
state = env.reset()
state = np.reshape(state, [1,4])

for episode in range(10):
    env.reset()
    for t in range(200):
        env.render()
        action = trained_model.action(state)
        nextstate, reward, done, info = env.step(action)
        next_state = np.reshape(nextstate, [1,4])
        state = next_state
        print(t, next_state, reward, done, info, action)
        if done:
            break
print('Done')


# Create GIF
# screen = env.render(mode='rgb_array')
# im = Image.fromarray(screen)
# images = [im]
#
# state = tf.constant(env.reset(), dtype=tf.float32)
#
# for t in range(200):
#     state = tf.expand_dims(state, 0)
#
#     screen = env.render()
#     action = trained_model.action(state)
#     state, reward, done, info = env.step(action)
#     state = tf.constant(state, dtype=tf.float32)
#     if t % 5 == 0:
#         screen = env.render(mode='rgb_array')
#         images.append(Image.fromarray(screen))
#
#     if done:
#         break
#
# image_file = 'cartpole.gif'
# images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)