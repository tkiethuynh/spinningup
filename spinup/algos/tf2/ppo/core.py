import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium.spaces import Box, Discrete

from spinup.algos.tf2._keras_utils import SerializableActorCritic


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(input_dim, hidden_sizes=(32,), activation='tanh', output_activation=None):
    model = tf.keras.Sequential([layers.Input(shape=(int(input_dim),))])
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=int(h), activation=activation))
    model.add(layers.Dense(units=int(hidden_sizes[-1]), activation=output_activation))
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
        return tf.reduce_sum(
            tf.one_hot(tf.cast(a, tf.int32), depth=logits.shape[-1]) * tf.nn.log_softmax(logits),
            axis=-1,
        )

    def entropy(self, logits):
        return tf.reduce_sum(-tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1)


class MLPGaussianActor(tf.keras.Model):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp(obs_dim, list(hidden_sizes) + [act_dim], activation)
        # State-independent log standard deviation vector, initialized to -0.5
        self.log_std = self.add_weight(
            name="log_std",
            shape=(act_dim,),
            initializer=tf.keras.initializers.Constant(-0.5),
            dtype="float32",
            trainable=True,
        )

    def call(self, obs):
        mu = self.mu_net(obs)
        std = tf.broadcast_to(tf.exp(self.log_std), tf.shape(mu))
        return mu, std

    def action_distribution(self, obs):
        mu, std = self.call(obs)
        return mu + tf.random.normal(tf.shape(mu)) * std

    def log_prob_from_distribution(self, mu_std, a):
        mu, std = mu_std
        log_std = tf.math.log(std)
        pre_sum = -0.5 * (((a - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)

    def entropy(self, mu_std):
        _, std = mu_std
        return tf.reduce_sum(tf.math.log(std) + 0.5 * np.log(2 * np.pi * np.e), axis=-1)


class MLPCritic(tf.keras.Model):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp(obs_dim, list(hidden_sizes) + [1], activation)

    def call(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)    # Critical to ensure v has right shape.


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ppo")
class MLPActorCritic(SerializableActorCritic):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self._save_init_spec(observation_space, action_space, hidden_sizes, activation)

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def call(self, obs):
        return self.pi(obs), self.v(obs)

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

    def act(self, obs, deterministic=False):
        if isinstance(self.pi, MLPCategoricalActor):
            logits = self.pi(obs)
            if deterministic:
                a = tf.argmax(logits, axis=-1)
            else:
                a = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1)
        else:
            mu, std = self.pi(obs)
            if deterministic:
                a = mu
            else:
                a = mu + tf.random.normal(tf.shape(mu)) * std
        return a.numpy()
