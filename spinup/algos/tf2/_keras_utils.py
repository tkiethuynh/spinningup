from gymnasium.spaces import Box, Discrete
import numpy as np
import tensorflow as tf


class SerializableActorCritic(tf.keras.Model):
    """
    Base Keras Model for Spinning Up actor-critics that enables ``.keras``
    saving/loading from Gym space specs without duplicating serialization
    boilerplate across every algorithm's ``core.py``.
    """

    def _save_init_spec(self, observation_space, action_space, hidden_sizes, activation):
        self._obs_shape = list(int(x) for x in observation_space.shape)
        self._is_discrete = isinstance(action_space, Discrete)
        self._act_n = int(action_space.n) if self._is_discrete else None
        self._act_shape = [] if self._is_discrete else list(int(x) for x in action_space.shape)
        self._act_limit = None if self._is_discrete else float(action_space.high[0])
        self._hidden_sizes = list(hidden_sizes)
        self._activation = tf.keras.activations.serialize(tf.keras.activations.get(activation))

    def get_config(self):
        config = super().get_config()
        config.update({
            "obs_shape": self._obs_shape,
            "is_discrete": self._is_discrete,
            "act_n": self._act_n,
            "act_shape": self._act_shape,
            "act_limit": self._act_limit,
            "hidden_sizes": self._hidden_sizes,
            "activation": self._activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        obs_shape = tuple(config.pop("obs_shape"))
        is_discrete = config.pop("is_discrete")
        act_n = config.pop("act_n")
        act_shape = tuple(config.pop("act_shape"))
        act_limit = config.pop("act_limit")
        obs_space = Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        if is_discrete:
            act_space = Discrete(act_n)
        else:
            act_space = Box(-act_limit, act_limit, shape=act_shape, dtype=np.float32)
        return cls(obs_space, act_space, **config)

    def build(self, input_shape):
        self.built = True
