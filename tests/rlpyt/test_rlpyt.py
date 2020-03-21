"""
requirements:
atari_py  ## pip install --user atari_py
cv2       ## conda install -c conda-forge opencv
psutil    ## pip install --user psutil
pyprind   ## pip install --user pyprind
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = AtariDqnAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


def test_rlpyt():
    # import argparse
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--game', help='Atari game', default='pong')
    # parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    # parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    # args = parser.parse_args()
    build_and_train(
        game='pong',
        run_ID=0,
        cuda_idx=None,
    )
