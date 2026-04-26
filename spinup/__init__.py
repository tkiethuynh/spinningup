# Setup GPU environment before any backend imports
from spinup.utils.device_utils import setup_tf_gpu
setup_tf_gpu()

# TF2 Algorithms
try:
    from spinup.algos.tf2.vpg.vpg import vpg as vpg_tf2
    from spinup.algos.tf2.ppo.ppo import ppo as ppo_tf2
    from spinup.algos.tf2.ddpg.ddpg import ddpg as ddpg_tf2
    from spinup.algos.tf2.td3.td3 import td3 as td3_tf2
    from spinup.algos.tf2.sac.sac import sac as sac_tf2
    from spinup.algos.tf2.trpo.trpo import trpo as trpo_tf2
except ImportError:
    pass

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
