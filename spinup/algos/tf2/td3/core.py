import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from spinup.algos.tf2._keras_utils import SerializableActorCritic


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


class MLPActor(tf.keras.Model):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.act_limit = float(act_limit)
        self.pi = mlp(obs_dim, list(hidden_sizes) + [act_dim], activation, 'tanh')

    def call(self, obs):
        # Scale tanh output from [-1, 1] to [-act_limit, act_limit]
        return self.act_limit * self.pi(obs)


class MLPQFunction(tf.keras.Model):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, list(hidden_sizes) + [1], activation, None)

    def call(self, obs, act):
        q = self.q(tf.concat([obs, act], axis=-1))
        return tf.squeeze(q, axis=-1)    # Critical to ensure q has right shape.


@tf.keras.utils.register_keras_serializable(package="spinup.tf2.td3")
class MLPActorCritic(SerializableActorCritic):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation='relu', **kwargs):
        super().__init__(**kwargs)
        self._save_init_spec(observation_space, action_space, hidden_sizes, activation)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and twin Q-functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def call(self, obs, act=None):
        pi_out = self.pi(obs)
        if act is not None:
            return pi_out, self.q1(obs, act), self.q2(obs, act)
        return pi_out, self.q1(obs, pi_out), self.q2(obs, pi_out)

    def act(self, obs, deterministic=False):
        return self.pi(obs).numpy()
