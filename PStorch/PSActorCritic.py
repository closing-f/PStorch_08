from Arguments import parse_args
from abc import ABC
from stable_baselines3.td3.policies import Actor
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import ContinuousCritic
from typing import Tuple


class PSActor(Actor, ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):

        super(PSActor, self).__init__(
            *args,
            **kwargs
        )

        args = parse_args()

        shared_net, policy_net = [], []
        policy_only_layers = []
        feature_dim = args.uav_obs_dim - 2
        last_layer_dim_shared = feature_dim
        for idx, layer in enumerate(args.net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(args.activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict)
                if "pi" in layer:
                    assert isinstance(layer["pi"], list)
                    policy_only_layers = layer["pi"]
                break  # From here on the network splits up in policy and value network

        self.uav_obs_dim = args.uav_obs_dim
        self.pi_add_dim = self.uav_obs_dim + 2 + (args.nb_PoIs * args.poi_dim)

        last_layer_dim_pi = last_layer_dim_shared + self.pi_add_dim
        for idx, pi_layer_size in enumerate(policy_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(args.activation_fn())
                last_layer_dim_pi = pi_layer_size

        self.action_dim = get_action_dim(self.action_space)
        policy_net.append(nn.Linear(last_layer_dim_pi, self.action_dim))
        policy_net.append(nn.Tanh())

        self.shared_net_pi = nn.Sequential(*shared_net)
        self.mu = nn.Sequential(*policy_net)

        self.nr_agents = args.nb_UAVs
        self.me_dim = args.me_dim
        self.max_nb_UAVs = args.max_nb_UAVs
        self.me_dim_single = (self.max_nb_UAVs - 1) * self.uav_obs_dim
        self.features_dim_all = self.me_dim_single + self.pi_add_dim


    def forward(self, features: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        if isinstance(features, np.ndarray):
            features = th.from_numpy(features).float()
        assert th.is_tensor(features), "Error, NN input is not tensor"
        features = features.float()
        features = th.reshape(features, (self.nr_agents, -1, self.features_dim_all))
        me_input = features[:, :, :self.me_dim_single]
        uav_self_input = features[:, :, self.me_dim_single:]
        batch_size = me_input.shape[1]
        me_data_input = th.reshape(me_input, (self.nr_agents,
                                              batch_size, self.max_nb_UAVs - 1, self.uav_obs_dim))
        me_valid_index = me_data_input[:, :, :, self.uav_obs_dim - 2]
        num_neighbor = th.sum(me_valid_index, dim=-1).flatten()
        shared_me_latent_pi = th.zeros((self.nr_agents, batch_size, self.me_dim)).cuda()
        for idx_UAV, batch_data_single_UAV in enumerate(me_data_input):
            me_tmp_input = batch_data_single_UAV[:, :int(num_neighbor[idx_UAV].item()), :self.uav_obs_dim - 2]
            shared_me_latent_idx_UAV_pi = self.shared_net_pi(me_tmp_input)
            shared_me_latent_idx_UAV_mean_pi = th.mean(shared_me_latent_idx_UAV_pi, dim=1)
            shared_me_latent_pi[idx_UAV, :] = shared_me_latent_idx_UAV_mean_pi
        me_output_pi = th.cat((shared_me_latent_pi, uav_self_input), -1)

        return self.mu(me_output_pi)


class PSCritic(ContinuousCritic):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(PSCritic, self).__init__(
            *args,
            **kwargs
        )

        args = parse_args()

        shared_net, value_net = [], []
        value_only_layers = []
        feature_dim = args.uav_obs_dim - 2
        last_layer_dim_shared = feature_dim
        for idx, layer in enumerate(args.net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(args.activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict)
                if "vf" in layer:
                    assert isinstance(layer["vf"], list)
                    policy_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        self.uav_obs_dim = args.uav_obs_dim
        self.vf_add_dim = self.uav_obs_dim + 2 + (args.nb_PoIs * args.poi_dim)
        self.action_dim = get_action_dim(self.action_space)

        last_layer_dim_vf = last_layer_dim_shared + self.vf_add_dim + self.action_dim
        for idx, vf_layer_size in enumerate(value_only_layers):
            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(args.activation_fn())
                last_layer_dim_vf = vf_layer_size

        value_net.append(nn.Linear(last_layer_dim_vf, 1))

        self.shared_net_vf = nn.Sequential(*shared_net)
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net = nn.Sequential(*value_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.nr_agents = args.nb_UAVs
        self.me_dim = args.me_dim
        self.max_nb_UAVs = args.max_nb_UAVs
        self.me_dim_single = (self.max_nb_UAVs - 1) * self.uav_obs_dim
        self.features_dim_all = self.me_dim_single + self.vf_add_dim

    def extract_features(self, features):

        if isinstance(features, np.ndarray):
            features = th.from_numpy(features).float()
        assert th.is_tensor(features), "Error, NN input is not tensor"

        features = th.reshape(features, (self.nr_agents, -1, self.features_dim_all))
        me_input = features[:, :, :self.me_dim_single]
        uav_self_input = features[:, :, self.me_dim_single:]
        batch_size = me_input.shape[1]
        me_data_input = th.reshape(me_input, (self.nr_agents,
                                              batch_size, self.max_nb_UAVs - 1, self.uav_obs_dim))
        me_valid_index = me_data_input[:, :, :, self.uav_obs_dim - 2]
        num_neighbor = th.sum(me_valid_index, dim=-1).flatten()
        shared_me_latent_vf = th.zeros((self.nr_agents, batch_size, self.me_dim)).cuda()
        for idx_UAV, batch_data_single_UAV in enumerate(me_data_input):
            me_tmp_input = batch_data_single_UAV[:, :int(num_neighbor[idx_UAV].item()), :self.uav_obs_dim - 2]
            shared_me_latent_idx_UAV_vf = self.shared_net_vf(me_tmp_input)
            shared_me_latent_idx_UAV_mean_vf = th.mean(shared_me_latent_idx_UAV_vf, dim=1)
            shared_me_latent_vf[idx_UAV, :] = shared_me_latent_idx_UAV_mean_vf
        me_output_vf = th.cat((shared_me_latent_vf, uav_self_input), -1)
        return me_output_vf

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:

        me_output_vf = self.extract_features(features=obs)

        # print("critic forward vfoutps shape: {}".format(me_output_vf.shape)) (6, 100, 572)
        # print("critic forward actions shape: {}".format(actions.shape))   (6, 100, 2)
        # assert me_output_vf.shape == actions.shape, "critic forward shape unequal"
        qvalue_input = th.cat([me_output_vf, actions], dim=-1)
        # print("cat shape: {}".format(qvalue_input.shape)) (6, 100, 574)

        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:

        features = self.extract_features(features=obs)
        return self.q_networks[0](th.cat([features, actions], dim=-1))