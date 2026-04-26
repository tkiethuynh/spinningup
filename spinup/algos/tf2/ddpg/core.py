import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def mlp(input_dim, hidden_sizes=(32,), activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=h, activation=activation))
    model.add(layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    return model

class MLPActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.pi = mlp(obs_dim, list(hidden_sizes) + [act_dim], activation, 'tanh')
        self.act_limit = act_limit

    def call(self, obs):
        return self.act_limit * self.pi(obs)

class MLPQFunction(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, list(hidden_sizes) + [1], activation, None)

    def call(self, obs, act):
        q = self.q(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q, axis=-1)

class MLPActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(256,256), activation='relu'):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q  = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def call(self, obs, act=None):
        pi_out = self.pi(obs)
        if act is not None:
            q_out = self.q(obs, act)
        else:
            q_out = self.q(obs, pi_out)
        return pi_out, q_out

    def act(self, obs):
        return self.pi(obs).numpy()
