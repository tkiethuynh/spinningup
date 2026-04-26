import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium.spaces import Box, Discrete

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(input_dim, hidden_sizes=(32,), activation='tanh', output_activation=None):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=h, activation=activation))
    model.add(layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    return model

class MLPCategoricalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(obs_dim, list(hidden_sizes) + [act_dim], activation)

    def call(self, obs):
        return self.logits_net(obs)

    def action_distribution(self, obs):
        logits = self.call(obs)
        return tf.random.categorical(logits, num_samples=1)

    def log_prob_from_distribution(self, logits, a):
        return tf.reduce_sum(tf.one_hot(a, depth=logits.shape[-1]) * tf.nn.log_softmax(logits), axis=1)

class MLPGaussianActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp(obs_dim, list(hidden_sizes) + [act_dim], activation)
        self.log_std = tf.Variable(initial_value=-0.5 * np.ones(act_dim, dtype=np.float32), trainable=True)

    def call(self, obs):
        mu = self.mu_net(obs)
        return mu, tf.exp(self.log_std)

    def action_distribution(self, obs):
        mu, std = self.call(obs)
        return mu + tf.random.normal(tf.shape(mu)) * std

    def log_prob_from_distribution(self, mu_std, a):
        mu, std = mu_std
        log_std = tf.math.log(std)
        pre_sum = -0.5 * (((a - mu) / (std + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

class MLPCritic(tf.keras.Model):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp(obs_dim, list(hidden_sizes) + [1], activation)

    def call(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)

class MLPActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation='tanh'):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def call(self, obs):
        pi_out = self.pi(obs)
        v_out = self.v(obs)
        return pi_out, v_out

    def step(self, obs):
        if isinstance(self.pi, MLPCategoricalActor):
            logits = self.pi(obs)
            a = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1)
            logp_a = self.pi.log_prob_from_distribution(logits, a)
        else:
            mu, std = self.pi(obs)
            a = mu + tf.random.normal(tf.shape(mu)) * std
            logp_a = self.pi.log_prob_from_distribution((mu, std), a)
        
        v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()
