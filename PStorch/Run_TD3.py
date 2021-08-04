from Arguments import parse_args
from PSTD3policy import PSTD3policy
from UAVEnv import UAVnet
import os
import random
import torch as th
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3 import DDPG


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    args = parse_args()
    setup_seed(args.seed)
    uav_env = UAVnet(arglist=args)
    net_arch = args.net_arch
    ac_fn = args.activation_fn
    model = DDPG(PSTD3policy, uav_env, policy_kwargs={"net_arch": net_arch, "activation_fn": ac_fn},
                buffer_size=int(3e4))
    model.learn(args.train_steps * args.train_episodes)
