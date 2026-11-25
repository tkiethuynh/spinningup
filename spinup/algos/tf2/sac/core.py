import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gymnasium.spaces import Box

EPS = 1e-6


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


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.sac")
class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))
        self.act_limit = float(act_limit)
        self.net = mlp(self.obs_dim, self.hidden_sizes, self.activation, self.activation)
        self.mu_layer = layers.Dense(self.act_dim)
        self.log_std_layer = layers.Dense(self.act_dim)
        feat_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.obs_dim
        self.mu_layer.build((None, feat_dim))
        self.log_std_layer.build((None, feat_dim))

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

    def call(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        if deterministic:
            pi_action = mu
        else:
            pi_action = mu + tf.random.normal(tf.shape(mu)) * std

        if with_logprob:
            logp_pi = tf.reduce_sum(
                -0.5 * (((pi_action - mu) / (std + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi)),
                axis=-1,
            )
            logp_pi -= tf.reduce_sum(
                2 * (np.log(2) - pi_action - tf.nn.softplus(-2 * pi_action)),
                axis=-1,
            )
        else:
            logp_pi = None

        pi_action = tf.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.sac")
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


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.sac")
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

        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, self.hidden_sizes, self.activation, self.act_limit
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, self.hidden_sizes, self.activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, self.hidden_sizes, self.activation)

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
        pi_out, _ = self.pi(obs)
        if act is not None:
            q1_out = self.q1(obs, act)
            q2_out = self.q2(obs, act)
        else:
            q1_out = self.q1(obs, pi_out)
            q2_out = self.q2(obs, pi_out)
        return pi_out, q1_out, q2_out

    def act(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
        return a.numpy()
