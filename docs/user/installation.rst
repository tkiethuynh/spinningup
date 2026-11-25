============
Installation
============


.. contents:: Table of Contents

Spinning Up requires Python3, Gymnasium, and OpenMPI. 

Spinning Up is currently only supported on Linux and OSX. 

.. admonition:: You Should Know

    Many examples and benchmarks in Spinning Up refer to RL environments that use the `MuJoCo`_ physics engine. MuJoCo is now free and open source. Installing it is optional, but because of its importance to the research community---it is the de facto standard for benchmarking deep RL algorithms in continuous control---it is preferred. 

    Don't worry if you decide not to install MuJoCo, though. You can definitely get started in RL by running RL algorithms on the `Classic Control`_ and `Box2d`_ environments in Gymnasium, which are totally free to use.

.. _`Classic Control`: https://gymnasium.farama.org/environments/classic_control/
.. _`Box2d`: https://gymnasium.farama.org/environments/box2d/
.. _`MuJoCo`: https://mujoco.org/

Installing Python
=================

We recommend installing Python through Anaconda. Anaconda is a library that includes Python and many useful packages for Python, as well as an environment manager called conda that makes package management simple.

Follow `the installation instructions`_ for Anaconda here. Download and install Anaconda3. Then create a conda Python 3.12 env (Python 3.10–3.13 supported) for organizing packages used in Spinning Up:

.. parsed-literal::

    conda create -n spinningup python=3.12

To use Python from the environment you just created, activate the environment with:

.. parsed-literal::

    conda activate spinningup

.. admonition:: You Should Know

    If you're new to python environments and package management, this stuff can quickly get confusing or overwhelming, and you'll probably hit some snags along the way. (Especially, you should expect problems like, "I just installed this thing, but it says it's not found when I try to use it!") You may want to read through some clean explanations about what package management is, why it's a good idea, and what commands you'll typically have to execute to correctly use it. 

    `FreeCodeCamp`_ has a good explanation worth reading. There's a shorter description on `Towards Data Science`_ which is also helpful and informative. Finally, if you're an extremely patient person, you may want to read the (dry, but very informative) `documentation page from Conda`_.

.. _`the installation instructions`: https://docs.continuum.io/anaconda/install/
.. _`Anaconda3-5.3.0`: https://repo.anaconda.com/archive/
.. _`FreeCodeCamp`: https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c
.. _`Towards Data Science`: https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097
.. _`documentation page from Conda`: https://conda.io/docs/user-guide/tasks/manage-environments.html
.. _`this Github issue for Tensorflow`: https://github.com/tensorflow/tensorflow/issues/20444


Installing OpenMPI
==================

Ubuntu 
------

.. parsed-literal::

    sudo apt-get update && sudo apt-get install -y openmpi-bin libopenmpi-dev swig python3-dev


Mac OS X
--------
Installation of system packages on Mac requires Homebrew_. With Homebrew installed, run the follwing:

.. parsed-literal::

    brew install openmpi swig

.. _Homebrew: https://brew.sh

Installing Spinning Up
======================

.. parsed-literal::

    git clone https://github.com/tkiethuynh/spinningup.git
    cd spinningup
    pip install -e .

.. admonition:: You Should Know

    Spinning Up installs Gymnasium Classic Control and open-source MuJoCo environments by default. To install Box2D (e.g., ``LunarLander-v3``) or all optional extras, run ``pip install -e ".[box2d]"`` or ``pip install -e ".[all]"``. Check out the `Gymnasium`_ github page for help.

.. _`Gymnasium`: https://github.com/Farama-Foundation/Gymnasium

Check Your Install
==================

To see if you've successfully installed Spinning Up, try running PPO in the ``CartPole-v1`` (or ``LunarLander-v3`` with ``.[box2d]``) environment with

.. parsed-literal::

    python -m spinup.run ppo --hid "[32,32]" --env CartPole-v1 --exp_name installtest --gamma 0.999

This runs quickly so you can see learning progress right away.

After it finishes training, watch a video of the trained policy with

.. parsed-literal::

    python -m spinup.run test_policy data/installtest/installtest_s0

And plot the results with

.. parsed-literal::

    python -m spinup.run plot data/installtest/installtest_s0


MuJoCo Environments
===================

Gymnasium's modern ``v5`` MuJoCo environments use the free, open-source `MuJoCo`_ Python bindings and are included automatically when installing Spinning Up.

Check that MuJoCo environments are working by running PPO in ``Walker2d-v5``:

.. parsed-literal::

    python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v5 --exp_name mujocotest
