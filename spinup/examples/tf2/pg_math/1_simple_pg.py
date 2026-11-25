import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from spinup.utils.device_utils import setup_tf_gpu
setup_tf_gpu()

import tensorflow as tf
from tensorflow.keras import layers


def mlp(obs_dim, sizes, activation='tanh', output_activation=None):
    # Build a feedforward neural network.
    model = tf.keras.Sequential([layers.Input(shape=(int(obs_dim),))])
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        model.add(layers.Dense(int(sizes[j + 1]), activation=act))
    return model


def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = int(env.observation_space.shape[0])
    n_acts = int(env.action_space.n)

    # make core of policy network
    logits_net = mlp(obs_dim, sizes=[obs_dim] + hidden_sizes + [n_acts])

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        logits = logits_net(obs.reshape(1, -1).astype(np.float32))
        return int(tf.random.categorical(logits, num_samples=1)[0, 0].numpy())

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logits = logits_net(obs)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        logp = tf.reduce_sum(tf.one_hot(act, depth=n_acts) * log_probs, axis=-1)
        return -tf.reduce_mean(logp * weights)

    # make optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()    # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(obs)
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                (obs, _), done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        obs_t = tf.convert_to_tensor(np.asarray(batch_obs), dtype=tf.float32)
        act_t = tf.convert_to_tensor(np.asarray(batch_acts), dtype=tf.int32)
        weights_t = tf.convert_to_tensor(batch_weights, dtype=tf.float32)
        with tf.GradientTape() as tape:
            batch_loss = compute_loss(obs_t, act_t, weights_t)
        grads = tape.gradient(batch_loss, logits_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, logits_net.trainable_variables))
        return float(batch_loss), batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient (TF2).\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
