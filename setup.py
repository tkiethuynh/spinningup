from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 10, \
    "The Spinning Up repo is designed to work with Python 3.10 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle',
        'gymnasium[atari,box2d,classic_control,mujoco]',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy==1.26.4',
        'pandas',
        'pytest',
        'psutil',
        'scipy==1.12.0',
        'seaborn',
        'tensorflow==2.15.0',
        'torch>=2.0',
        'tqdm',
        'six',
        'pyparsing',
        'pytz',
        'pyyaml',
        'distro',
        'google-genai'
    ],
    description="Modernized version of Spinning Up in Deep RL with TF2, PyTorch 2.x, and Gymnasium.",
    author="OpenAI & Kiet Huynh",
    url="https://github.com/tkiethuynh/spinningup",
)
