"""
Simple policy gradient in Keras
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils

tf.compat.v1.disable_eager_execution()


class Agent(object):
    def __init__(self, input_dim, output_dim):
        # We will be making custom NN pipelines for policy gradient as we dont have a loss function which we have to
        # minimize rather an objective function which we wnat to maximize
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        net = layers.Dense(100)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(50)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)
        self.model = Model(inputs=self.X, outputs=net)
        # this is our model whicih takes in self.X as input (having shape as (input_dims,)) and gives net (whose
        # procedure is explained in above lines)
        action_prob_placeholder = self.model.output
        # the output is a softmax latyer of actions probability

        # placeholders are variables (in form of tensors) which can be used in the computation graphs

        # action_onehot_placeholder is the variable to store which actions were taken in the trajectory and update them
        # and use them to calculate grads to update NN
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        # dicount_reward_placeholder is the variable to store the discounted rewards which will be used to calulate
        # grads for the NN
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")
        # findinng the probability of the action taken in that time step n the trajectory
        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)
        # this is the function we have minimize (its actually maximize thats why the negative sign)
        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)
        adam = Adam()
        # defining the update process, params tells which all parameters to update , loss is the the loss function
        # which is to be minimized
        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)
        # definig the train function - take self.model.input as input and action_hot_placeholder,
        # discount_onehot_placeholder as input and then call the Adam optimizer to calculte the loss from the
        # procedure explained above and the output of the model which is to be trained is self.model.outputs get
        # updates from updates variable above
        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[self.model.outputs],
                                   updates=updates)

    def get_action(self, state):
        shape = state.shape
        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

    def fit(self, S, action_list, reward_list):
        # action_list is the list of actions taken in the episode
        # converting it to one hot vector
        action_onehot = np_utils.to_categorical(action_list, num_classes=self.output_dim)
        discount_reward = compute_discounted_R(reward_list)
        # as defined above it takes in states,action_hot,discount_reward
        self.train_fn([S, action_onehot, discount_reward])


def compute_discounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    discounted_r -= (discounted_r.mean() / discounted_r.std())
    return discounted_r


def run_episode(env, agent):
    done = False
    state_list = []
    action_list = []
    reward_list = []
    curr_state = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent.get_action(np.asarray([curr_state]))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state_list.append(curr_state)
        action_list.append(action)
        reward_list.append(reward)
        curr_state = next_state

        if done:
            state_list = np.array(state_list)
            action_list = np.array(action_list)
            reward_list = np.array(reward_list)

            agent.fit(state_list, action_list, reward_list)

    return total_reward


def main():
    env = gym.make("Assault-ram-v0")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = Agent(input_dim, output_dim)
    for episode in range(2000):
        reward = run_episode(env, agent)
        print(episode, reward)


if __name__ == '__main__':
    main()
