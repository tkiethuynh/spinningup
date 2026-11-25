import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(input_dim, hidden_sizes=(32,), activation='tanh', output_activation=None):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=h, activation=activation))
    model.add(layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    return model


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ppo")
class MLPCategoricalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.logits_net = mlp(self.obs_dim, self.hidden_sizes + [self.act_dim], self.activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        self.built = True

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


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ppo")
class MLPGaussianActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.mu_net = mlp(self.obs_dim, self.hidden_sizes + [self.act_dim], self.activation)
        self.log_std = self.add_weight(
            name="log_std",
            shape=(self.act_dim,),
            initializer=tf.keras.initializers.Constant(-0.5),
            dtype="float32",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        self.built = True

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


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ppo")
class MLPCritic(tf.keras.Model):
    def __init__(self, obs_dim, hidden_sizes, activation, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.v_net = mlp(self.obs_dim, self.hidden_sizes + [1], self.activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_dim": self.obs_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        self.built = True

    def call(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ppo")
class MLPActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.obs_shape = tuple(int(x) for x in observation_space.shape)
        self.is_discrete = isinstance(action_space, Discrete)
        if self.is_discrete:
            self.act_n = int(action_space.n)
            self.act_shape = ()
        else:
            self.act_n = None
            self.act_shape = tuple(int(x) for x in action_space.shape)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))

        obs_dim = self.obs_shape[0]
        if self.is_discrete:
            self.pi = MLPCategoricalActor(obs_dim, self.act_n, self.hidden_sizes, self.activation)
        else:
            self.pi = MLPGaussianActor(obs_dim, self.act_shape[0], self.hidden_sizes, self.activation)

        self.v = MLPCritic(obs_dim, self.hidden_sizes, self.activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_shape": list(self.obs_shape),
            "is_discrete": self.is_discrete,
            "act_n": self.act_n,
            "act_shape": list(self.act_shape),
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        obs_shape = tuple(config.pop("obs_shape"))
        is_discrete = config.pop("is_discrete")
        act_n = config.pop("act_n")
        act_shape = tuple(config.pop("act_shape"))
        obs_space = Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        if is_discrete:
            act_space = Discrete(act_n)
        else:
            act_space = Box(-1.0, 1.0, shape=act_shape, dtype=np.float32)
        return cls(obs_space, act_space, **config)

    def build(self, input_shape):
        self.built = True

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
