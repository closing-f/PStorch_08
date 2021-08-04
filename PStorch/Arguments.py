import time
import torch
from torch import nn
import argparse
from typing import Dict, List, Tuple, Type, Union

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

def parse_args():
    parser = argparse.ArgumentParser("PS-based RL experiments for multi-UAV networks")

    # Model save and load parameters
    parser.add_argument("--Model", type=str, default='A2C', help="PPO A2C TD3 DDPG")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--reward_form", type=str, default='IA', help="LT PC IA RE AL")
    parser.add_argument("--path_file", type=str, default="C:/Users/user/Desktop/PStorch/save/20201223",
                        help="model save path")

    # inequality
    parser.add_argument("--beta", type=float, default=0.05, help="base is 0.05")
    parser.add_argument("--alpha", type=float, default=5, help="base is 5")

    # UAV_environment
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--nb_UAVs", type=int, default=6, help="the number of UAV agents")
    parser.add_argument("--comm_radius", type=float, default=500, help="communication range")
    parser.add_argument("--obs_radius", type=float, default=250, help="coverage range")
    # Fixed UAV parameters
    parser.add_argument("--torus", type=bool, default=False, help="If True, the world is cycle; Fixed para")
    parser.add_argument("--max_nb_UAVs", type=int, default=8, help="the max number of UAV agents; Fixed para")
    parser.add_argument("--uav_obs_dim", type=int, default=6,
                        help="dist, cos, sin, energy, pursuer(bool), all(bool); Fixed para")
    parser.add_argument("--nb_PoIs", type=int, default=100, help="the number of PoIs; Fixed para")
    parser.add_argument("--poi_dim", type=int, default=5, help="dist, cos, sin, cover(bool), c_time; Fixed para")
    parser.add_argument("--dim_a", type=int, default=2, help="v, w; Fixed para")
    parser.add_argument("--n_step", type=int, default=10240, help="the time of mission time; Fixed para")
    parser.add_argument("--world_size", type=float, default=1000, help="The world is cube; Fixed para")
    # base.py -> action_repeat = 10; self.dt = 0.01; self.max_lin_v = 50; self.max_ang_v = np.pi
    # base.py -> energy_fun = (1/3.0*v**3 - 0.0625*v + 0.03)/30=0.002
    # UAV.py -> self.energy = 3; self.dim_rec_o = (8, 6); self.dim_evader_o = (self.n_evaders, 5);
    # UAV.py -> self.dim_local_o = 2
    # Other UAV parameters
    parser.add_argument("--obs_mode", type=str, default='sum_obs_no_ori')
    parser.add_argument("--dynamics", type=str, default='unicycle')
    parser.add_argument("--distance_bins", type=int, default=8)
    parser.add_argument("--bearing_bins", type=int, default=8)

    # Parameters Sharing OnPolicy Algorithm
    parser.add_argument("--pi_dim", type=int, default=64, help="the dim of pi_only_network ouput")
    parser.add_argument("--vf_dim", type=int, default=64, help="the dim of vf_only_network output")
    parser.add_argument("--me_dim", type=int, default=64, help="this number is equal to net_arch's last int")
    parser.add_argument("--net_arch", type=List[Union[int, Dict[str, List[int]]]],
                        default=[64, 64, dict(vf=[64, 64], pi=[64, 64])], help="net_arch")
    parser.add_argument("--activation_fn", type=Type[nn.Module],
                        default=nn.Tanh, help="in SB3, default is nn.Tanh")

    # Train steps
    parser.add_argument("--train_steps", type=int, default=10240,
                        help="the train_steps of once model update， UAV is 1024 * 10")
    parser.add_argument("--train_episodes", type=int, default=40, help="Train episodes / 10")

    # evaluation parameters
    parser.add_argument("--episodes", type=int, default=3, help="the number of eval_episodes")
    parser.add_argument("--test_model_path_file", type=str, default="C:/Users/user/Desktop/PStorch/save/PPO/20201214",
                        help="model load path")

    return parser.parse_args()
