import os
import random
import torch as th
import numpy as np
from Arguments import parse_args
from UAVEnv import UAVnet
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    arglist = parse_args()
    arglist.nb_UAVs = 6
    setup_seed(1)
    uav_env = UAVnet(arglist=arglist)
    net_arch = arglist.net_arch
    ac_fn = arglist.activation_fn
    Load_Number = 3
    load_path = os.path.join(arglist.path_file, arglist.Model, arglist.reward_form,
                             'seed_{}'.format(arglist.seed), '{}.zip'.format(Load_Number))
    M = arglist.Model
    assert os.path.exists(load_path), "Load model is not existing"
    if M == 'PPO':
        model = PPO.load(path=load_path, env=uav_env)
    if M == 'A2C':
        model = A2C.load(path=load_path, env=uav_env)
    if M == 'TD3':
        model = TD3.load(path=load_path, env=uav_env)

    # model.policy.mlp_extractor.nr_agents = arglist.nb_UAVs

    obs = uav_env.reset()
    for i in range(1024):
        print(i)
        action, _state = model.predict(obs, deterministic=True)  # the state is defined for RNN
        # action = np.ones((6, 2))
        print("action: {}".format(action.flatten()))
        obs, reward, done, info = uav_env.step(action)
        print("reward: {}".format(reward.flatten()))
        uav_env.render()