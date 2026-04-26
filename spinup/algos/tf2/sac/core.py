import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

EPS = 1e-6

def mlp(input_dim, hidden_sizes=(32,), activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=h, activation=activation))
    model.add(layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    return model

class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.act_limit = act_limit
        self.net = mlp(obs_dim, list(hidden_sizes), activation, activation)
        self.mu_layer = layers.Dense(act_dim)
        self.log_std_layer = layers.Dense(act_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "act_limit": self.act_limit,
        })
        return config

    def call(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        # Pre-squash distribution and sample
        if deterministic:
            pi_action = mu
        else:
            pi_action = mu + tf.random.normal(tf.shape(mu)) * std

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        if with_logprob:
            logp_pi = tf.reduce_sum(-0.5 * (((pi_action - mu) / (std + EPS))**2 + 2 * log_std + np.log(2 * np.pi)), axis=-1)
            logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi_action - tf.nn.softplus(-2 * pi_action)), axis=-1)
        else:
            logp_pi = None

        pi_action = tf.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def entropy(self, mu, std):
        return tf.reduce_sum(tf.math.log(std) + 0.5 * np.log(2 * np.pi * np.e), axis=-1)

class MLPQFunction(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.q = mlp(obs_dim + act_dim, list(hidden_sizes) + [1], activation, None)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    def call(self, obs, act):
        q = self.q(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q, axis=-1)

class MLPActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(256,256), activation='relu'):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    def call(self, obs, act=None):
        pi_out, _ = self.pi(obs)
        if act is not None:
            q1_out = self.q1(obs, act)
            q2_out = self.q2(obs, act)
        else:
            q1_out = self.q1(obs, pi_out)
            q2_out = self.q2(obs, pi_out)
        return pi_out, q1_out, q2_out

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic, False)
        return a.numpy()
