import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium.spaces import Box


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(input_dim, hidden_sizes=(32,), activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_sizes[:-1]:
        model.add(layers.Dense(units=h, activation=activation))
    model.add(layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    return model


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ddpg")
class MLPActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.act_limit = float(act_limit)
        self.pi = mlp(self.obs_dim, self.hidden_sizes + [self.act_dim], self.activation, 'tanh')

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

    def build(self, input_shape):
        self.built = True

    def call(self, obs):
        return self.act_limit * self.pi(obs)


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ddpg")
class MLPQFunction(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.q = mlp(self.obs_dim + self.act_dim, self.hidden_sizes + [1], self.activation, None)

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

    def call(self, obs, act):
        q = self.q(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q, axis=-1)


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.ddpg")
class MLPActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.obs_shape = tuple(int(x) for x in observation_space.shape)
        self.act_shape = tuple(int(x) for x in action_space.shape)
        self.act_limit = float(action_space.high[0])
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))

        obs_dim = self.obs_shape[0]
        act_dim = self.act_shape[0]

        self.pi = MLPActor(obs_dim, act_dim, self.hidden_sizes, self.activation, self.act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, self.hidden_sizes, self.activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_shape": list(self.obs_shape),
            "act_shape": list(self.act_shape),
            "act_limit": self.act_limit,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        obs_shape = tuple(config.pop("obs_shape"))
        act_shape = tuple(config.pop("act_shape"))
        act_limit = float(config.pop("act_limit"))
        obs_space = Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        act_space = Box(-act_limit, act_limit, shape=act_shape, dtype=np.float32)
        return cls(obs_space, act_space, **config)

    def build(self, input_shape):
        self.built = True

    def call(self, obs, act=None):
        pi_out = self.pi(obs)
        if act is not None:
            q_out = self.q(obs, act)
        else:
            q_out = self.q(obs, pi_out)
        return pi_out, q_out

    def act(self, obs, deterministic=False):
        return self.pi(obs).numpy()
