======
Logger
======

.. contents:: Table of Contents

Using a Logger
==============

Spinning Up ships with basic logging tools, implemented in the classes `Logger`_ and `EpochLogger`_. The Logger class contains most of the basic functionality for saving diagnostics, hyperparameter configurations, the state of a training run, and the trained model. The EpochLogger class adds a thin layer on top of that to make it easy to track the average, standard deviation, min, and max value of a diagnostic over each epoch and across MPI workers.

.. admonition:: You Should Know

    All Spinning Up algorithm implementations use an EpochLogger.

.. _`Logger`: ../utils/logger.html#spinup.utils.logx.Logger
.. _`EpochLogger`: ../utils/logger.html#spinup.utils.logx.EpochLogger

Examples
--------

First, let's look at a simple example of how an EpochLogger keeps track of a diagnostic value:

>>> from spinup.utils.logx import EpochLogger
>>> epoch_logger = EpochLogger()
>>> for i in range(10):
        epoch_logger.store(Test=i)
>>> epoch_logger.log_tabular('Test', with_min_and_max=True)
>>> epoch_logger.dump_tabular()
-------------------------------------
|     AverageTest |             4.5 |
|         StdTest |            2.87 |
|         MaxTest |               9 |
|         MinTest |               0 |
-------------------------------------

The ``store`` method is used to save all values of ``Test`` to the ``epoch_logger``'s internal state. Then, when ``log_tabular`` is called, it computes the average, standard deviation, min, and max of ``Test`` over all of the values in the internal state. The internal state is wiped clean after the call to ``log_tabular`` (to prevent leakage into the statistics at the next epoch). Finally, ``dump_tabular`` is called to write the diagnostics to file and to stdout.

Next, let's look at a full training procedure with the logger embedded, to highlight configuration and model saving as well as diagnostic logging:

.. code-block:: python
   :linenos:

    import numpy as np
    import tensorflow as tf
    import time
    from spinup.utils.logx import EpochLogger

    # Simple script for training an MLP on MNIST with TF2.
    def train_mnist(steps_per_epoch=100, epochs=5,
                    lr=1e-3, layers=2, hidden_size=64,
                    logger_kwargs=dict(), save_freq=1):

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0

        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(28*28,))]
            + [tf.keras.layers.Dense(hidden_size, activation='relu') for _ in range(layers)]
            + [tf.keras.layers.Dense(10)]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        logger.setup_tf2_saver(model)
        start_time = time.time()

        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                idxs = np.random.randint(0, len(x_train), 32)
                xb, yb = x_train[idxs], y_train[idxs]
                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    loss = loss_fn(yb, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                acc = tf.reduce_mean(tf.cast(tf.equal(yb, tf.argmax(logits, axis=1)), tf.float32))
                logger.store(Loss=float(loss), Acc=float(acc))

            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state(state_dict=dict(), itr=None)

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Acc', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    if __name__ == '__main__':
        train_mnist()

In this example, observe that

* `logger.save_config`_ is used to save the hyperparameter configuration to a JSON file.
* `logger.setup_tf2_saver`_ is used to prepare the logger to save the Keras model.
* Diagnostics are saved to the logger's internal state via `logger.store`_.
* The model is saved once per epoch via `logger.save_state`_.
* `logger.log_tabular`_ and `logger.dump_tabular`_ are used to write the epoch diagnostics to file. Note that the keys passed into `logger.log_tabular`_ are the same as the keys passed into `logger.store`_.

.. _`logger.save_config`: ../utils/logger.html#spinup.utils.logx.Logger.save_config
.. _`logger.setup_tf2_saver`: ../utils/logger.html#spinup.utils.logx.Logger.setup_tf2_saver
.. _`logger.store`: ../utils/logger.html#spinup.utils.logx.EpochLogger.store
.. _`logger.save_state`: ../utils/logger.html#spinup.utils.logx.Logger.save_state
.. _`logger.log_tabular`: ../utils/logger.html#spinup.utils.logx.EpochLogger.log_tabular
.. _`logger.dump_tabular`: ../utils/logger.html#spinup.utils.logx.Logger.dump_tabular


Logging and PyTorch
-------------------

The preceding example was given in Tensorflow 2. For PyTorch, instead of ``logger.setup_tf2_saver``, you would use ``logger.setup_pytorch_saver``, and pass it `a PyTorch module`_ (the network you are training) as an argument.

The behavior of ``logger.save_state`` is the same as in the Tensorflow case: each time it is called, it'll save the latest version of the PyTorch module.

.. _`a PyTorch module`: https://pytorch.org/docs/stable/nn.html#torch.nn.Module

Logging and MPI
---------------

.. admonition:: You Should Know

    Several algorithms in RL are easily parallelized by using MPI to average gradients and/or other key quantities. The Spinning Up loggers are designed to be well-behaved when using MPI: things will only get written to stdout and to file from the process with rank 0. But information from other processes isn't lost if you use the EpochLogger: everything which is passed into EpochLogger via ``store``, regardless of which process it's stored in, gets used to compute average/std/min/max values for a diagnostic.


Logger Classes
==============


.. autoclass:: spinup.utils.logx.Logger
    :members:

    .. automethod:: spinup.utils.logx.Logger.__init__

.. autoclass:: spinup.utils.logx.EpochLogger
    :show-inheritance:
    :members:


Loading Saved Models (PyTorch Only)
===================================

To load an actor-critic model saved by a PyTorch Spinning Up implementation, run:

.. code-block:: python

    ac = torch.load('path/to/model.pt', weights_only=False)

When you use this method to load an actor-critic model, you can minimally expect it to have an ``act`` method that allows you to sample actions from the policy, given observations:

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Loading Saved Models (Tensorflow 2 Only)
========================================

.. autofunction:: spinup.utils.logx.restore_tf2_model

When you use this method to restore a model saved by a Tensorflow 2 Spinning Up implementation, it returns an ``MLPActorCritic`` Keras model with an ``act`` method and policy/value submodules (e.g., ``pi``, ``v``, ``q``, ``q1``, ``q2``). See each algorithm's documentation page for details.