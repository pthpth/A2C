import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


class Agent(object):
    def __init__(self, input_dim, output_dim):
        # We will be making custom NN pipelines for policy gradient as we dont have a loss function which we have to
        # minimize rather an objective function which we wnat to maximize
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = layers.Input(shape=(input_dim,))
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        critic_net = self.X
        critic_net = layers.Dense(40,kernel_initializer=initializer)(critic_net)
        critic_net = layers.Activation("relu")(critic_net)
        critic_net = layers.Dense(10,kernel_initializer=initializer)(critic_net)
        critic_net = layers.Activation("relu")(critic_net)
        critic_net = layers.Dense(1,kernel_initializer=initializer)(critic_net)

        self.critic_model = Model(inputs=self.X, outputs=critic_net)

        actor_net = self.X
        actor_net = layers.Dense(40,kernel_initializer=initializer)(actor_net)
        actor_net = layers.Activation("relu")(actor_net)
        actor_net = layers.Dense(20,kernel_initializer=initializer)(actor_net)
        actor_net = layers.Activation("relu")(actor_net)
        actor_net = layers.Dense(output_dim,kernel_initializer=initializer)(actor_net)
        actor_net = layers.Activation("softmax")(actor_net)

        self.actor_model = Model(inputs=self.X, outputs=actor_net)
        # this is our model whicih takes in self.X as input (having shape as (input_dims,)) and gives net (whose
        # procedure is explained in above lines)
        entropy_placeholder = K.placeholder(shape=(None,), name="entropy")
        action_prob_placeholder = self.actor_model.output
        value_placeholder = self.critic_model.output
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

        advantage_placeholder = discount_reward_placeholder - value_placeholder
        # findinng the probability of the action taken in that time step n the trajectory
        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)
        entropy_factor=0.005
        # this is the function we have minimize (its actually maximize thats why the negative sign)
        actor_loss = - log_action_prob * advantage_placeholder
        actor_loss = K.mean(actor_loss-entropy_factor * entropy_placeholder)
        adam = Adam(learning_rate=0.001)

        critic_loss = advantage_placeholder ** 2 * 0.5
        critic_loss = K.mean(critic_loss)
        adam2 = Adam(learning_rate=0.001)
        # defining the update process, params tells which all parameters to update , loss is the the loss function
        # which is to be minimized
        actor_updates = adam.get_updates(params=self.actor_model.trainable_weights, loss=actor_loss)
        critic_updates = adam2.get_updates(params=self.critic_model.trainable_weights, loss=critic_loss)
        # definig the train function - take self.model.input as input and action_hot_placeholder,
        # discount_onehot_placeholder as input and then call the Adam optimizer to calculte the loss from the
        # procedure explained above and the output of the model which is to be trained is self.model.outputs get
        # updates from updates variable above
        self.actor_train_fn = K.function(inputs=[self.actor_model.input,
                                                 action_onehot_placeholder,
                                                 discount_reward_placeholder, entropy_placeholder],
                                         outputs=[self.actor_model.outputs],
                                         updates=actor_updates)
        self.critic_train_fn = K.function(
            inputs=[self.critic_model.input, discount_reward_placeholder, entropy_placeholder],
            outputs=[self.critic_model.outputs], updates=critic_updates)

    def get_action(self, state):
        shape = state.shape
        action_prob = np.squeeze(self.actor_model.predict(state))
        # print(action_prob)
        state_value = np.squeeze(self.critic_model.predict(state))
        # print(state_value)
        return np.random.choice(np.arange(self.output_dim), p=action_prob), -np.sum(
            np.mean(action_prob) * np.log(action_prob+1e-12))

    def fit(self, state_list, action_list, reward_list, entropy):
        # action_list is the list of actions taken in the episode
        # converting it to one hot vector
        action_onehot = np_utils.to_categorical(action_list, num_classes=self.output_dim)
        discounted_r = compute_dicounted_R(reward_list)
        # as defined above it takes in states,action_hot,discount_reward
        print(entropy)
        self.actor_train_fn([state_list, action_onehot, discounted_r, entropy])
        self.critic_train_fn([state_list, discounted_r, entropy])


def compute_dicounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    discounted_r = (discounted_r - discounted_r.mean()) / (discounted_r.std()+1e-18)
    return discounted_r


def run_episode(env, agent):
    entropy = 0
    done = False
    state_list = []
    action_list = []
    reward_list = []
    curr_state = env.reset()
    total_reward = 0
    while not done:
        # print(agent.critic_model.layers[0].weights[0])
        action, entropy_curr = agent.get_action(np.asarray([curr_state]))
        entropy += entropy_curr
        # print(entropy)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # print(action,)
        state_list.append(curr_state)
        action_list.append(action)
        reward_list.append(reward)
        curr_state = next_state

        if done:
            state_list = np.array(state_list)
            action_list = np.array(action_list)
            reward_list = np.array(reward_list)
            if total_reward != 0:
                agent.fit(state_list, action_list, reward_list, entropy)
    return total_reward


def main():
    env = gym.make("Assault-v0")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = Agent(input_dim, output_dim)
    rewards = []
    for episode in range(1000):
        env.render()
        reward = run_episode(env, agent)
        rewards.append(reward)
        print(episode, reward)
    env.close()
    plt.plot(range(len(rewards)), rewards)
    plt.savefig("A2C.png")


if __name__ == '__main__':
    main()
