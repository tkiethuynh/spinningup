import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from spinup.algos.tf2._keras_utils import SerializableActorCritic

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(input_dim, hidden_sizes=(32,), activation='relu', output_activation=None):
    model = tf.keras.Sequential([layers.Input(shape=(int(input_dim),))])
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=int(h), activation=activation))
    model.add(layers.Dense(units=int(hidden_sizes[-1]), activation=output_activation))
    return model


class SquashedGaussianMLPActor(tf.keras.Model):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.act_limit = float(act_limit)
        self.net = mlp(obs_dim, list(hidden_sizes), activation, activation)
        self.mu_layer = layers.Dense(act_dim)
        self.log_std_layer = layers.Dense(act_dim)
        # Pre-build output heads so Keras 3 deserialization restores their weights cleanly
        feat_dim = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.mu_layer.build((None, feat_dim))
        self.log_std_layer.build((None, feat_dim))

    def call(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = tf.clip_by_value(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        # Pre-squash Gaussian sample (reparameterized via noise addition)
        pi_action = mu if deterministic else mu + tf.random.normal(tf.shape(mu)) * std

        if with_logprob:
            logp_pi = tf.reduce_sum(
                -0.5 * (((pi_action - mu) / (std + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi)),
                axis=-1,
            )
            # Enforcing Action Bounds (see Spinning Up SAC docs for tanh Jacobian correction)
            logp_pi -= tf.reduce_sum(
                2 * (np.log(2) - pi_action - tf.nn.softplus(-2 * pi_action)),
                axis=-1,
            )
        else:
            logp_pi = None

        pi_action = tf.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class MLPQFunction(tf.keras.Model):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, list(hidden_sizes) + [1], activation, None)

    def call(self, obs, act):
        q = self.q(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q, axis=-1)    # Critical to ensure q has right shape.


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.sac")
class MLPActorCritic(SerializableActorCritic):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation='relu', **kwargs):
        super().__init__(**kwargs)
        self._save_init_spec(observation_space, action_space, hidden_sizes, activation)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and twin Q-functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def call(self, obs, act=None):
        pi_out, _ = self.pi(obs)
        if act is not None:
            return pi_out, self.q1(obs, act), self.q2(obs, act)
        return pi_out, self.q1(obs, pi_out), self.q2(obs, pi_out)

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
        return a.numpy()
