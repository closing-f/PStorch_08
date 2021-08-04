import numpy as np
import torch as th
from torch import nn
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Any

# mean_embedding
# PS is parameters sharing
class PSNetwork(nn.Module):
    # CustomNetwork 直接写固定
    def __init__(
            self,
            arglist
    ):
        super(PSNetwork, self).__init__()
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []
        value_only_layers = []
        # only related last input dim
        # for UAV, just dist, cos, sin, energy
        feature_dim = arglist.uav_obs_dim - 2
        last_layer_dim_shared = feature_dim
        # build the network
        for idx, layer in enumerate(arglist.net_arch):
            # print(idx, layer)
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(arglist.activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict)
                if "pi" in layer:
                    assert isinstance(layer["pi"], list)
                    policy_only_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(layer["vf"], list)
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # other hyper-parameter
        self.max_nb_UAVs = arglist.max_nb_UAVs
        self.nr_agents = arglist.nb_UAVs
        # 6
        self.uav_obs_dim = arglist.uav_obs_dim
        self.me_dim = arglist.me_dim
        self.me_dim_single = (self.max_nb_UAVs - 1) * self.uav_obs_dim
        self.vfpi_add_dim = self.uav_obs_dim + 2 + (arglist.nb_PoIs * arglist.poi_dim)
        self.features_dim_all = self.me_dim_single + self.vfpi_add_dim

        last_layer_dim_pi = last_layer_dim_shared + self.vfpi_add_dim
        last_layer_dim_vf = last_layer_dim_shared + self.vfpi_add_dim
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(arglist.activation_fn())
                last_layer_dim_pi = pi_layer_size
            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(arglist.activation_fn())
                last_layer_dim_vf = vf_layer_size

        # IMPORTANT:
        # dim is feature，not output，vf_out_dim is 1, pi_out_dim depend on ac_space
        # mean_embedding network
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # create networks
        self.shared_net_pi = nn.Sequential(*shared_net)
        self.shared_net_vf = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

        # print(self.shared_net)
        # print(self.value_net)
        # print(self.policy_net)

    # : and -> is function describe
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # input dim is [nr_agent, sum_nei_obs(max_nr, uav_obs)+2+uav_obs]
        if isinstance(features, np.ndarray):
            # print("input is np.ndarray")
            features = th.from_numpy(features).float()
        # else:
        #     # print("input is tensor")
        assert th.is_tensor(features), "Error, NN input is not tensor"
        # print(features)
        # 起始位置（1，第二个元素）:结束位置(1, 第一个元素)
        # print(features)
        # 需要考虑 batch_size [nr_agents, batch_size, features_dim_all]
        features = th.reshape(features, (self.nr_agents, -1, self.features_dim_all))
        # print(features.shape)
        me_input = features[:, :, :self.me_dim_single]
        # print(me_input.shape)
        uav_self_input = features[:, :, self.me_dim_single:]
        # print(uav_self_input.shape)
        # print(me_input)
        # print(uav_self_input)
        batch_size = me_input.shape[1]
        me_data_input = th.reshape(me_input, (self.nr_agents,
                                              batch_size, self.max_nb_UAVs - 1, self.uav_obs_dim))
        # print(me_data_input.shape)
        # 就是用于判断哪些是邻居UAV的column
        me_valid_index = me_data_input[:, :, :, self.uav_obs_dim - 2]
        # print(me_valid_index.shape)
        # print(me_valid_index)
        num_neighbor = th.sum(me_valid_index, dim=-1).flatten()
        # print(num_neighbor.shape)
        '''
        防止0邻居的情况
        ones = th.ones_like(num_neighbor)
        nozero_num_neighbor = th.where(num_neighbor < 1, ones, num_neighbor)
        print(num_neighbor)
        print(nozero_num_neighbor)
        '''
        # In get_observation function in UAV.py, the data is queue
        # 现在根据 cover（bool）开始算mean
        shared_me_latent_pi = th.zeros((self.nr_agents, batch_size, self.me_dim)).cuda()
        shared_me_latent_vf = th.zeros((self.nr_agents, batch_size, self.me_dim)).cuda()
        # print(shared_me_latent.shape)
        for idx_UAV, batch_data_single_UAV in enumerate(me_data_input):
            # print(idx_UAV)
            # print(batch_data_single_UAV.shape)
            if num_neighbor[idx_UAV] < 1:
                # print("No UAV in communication range")
                pass
            else:
                # print(num_neighbor[idx_UAV])
                # print(num_neighbor[idx_UAV])
                # print(int(num_neighbor[idx_UAV].item()))
                # me_tmp_input[batch_size, neighbor_num, obs_dim]
                me_tmp_input = batch_data_single_UAV[:, :int(num_neighbor[idx_UAV].item()), :self.uav_obs_dim - 2]
                # bug: tensors 作为索引比喻是long，byte or bool
                # print(me_tmp_input.shape)
                # 此时已经get了idx_UAV的个体信息
                shared_me_latent_idx_UAV_pi = self.shared_net_pi(me_tmp_input)
                shared_me_latent_idx_UAV_vf = self.shared_net_vf(me_tmp_input)
                # print(shared_me_latent_idx_UAV.shape)
                shared_me_latent_idx_UAV_mean_pi = th.mean(shared_me_latent_idx_UAV_pi, dim=1)
                shared_me_latent_idx_UAV_mean_vf = th.mean(shared_me_latent_idx_UAV_vf, dim=1)

                # print(shared_me_latent_idx_UAV_mean.shape)
                shared_me_latent_pi[idx_UAV, :] = shared_me_latent_idx_UAV_mean_pi
                shared_me_latent_vf[idx_UAV, :] = shared_me_latent_idx_UAV_mean_vf
                # print(shared_me_latent.shape)
        me_output_pi = th.cat((shared_me_latent_pi, uav_self_input), -1)
        me_output_vf = th.cat((shared_me_latent_vf, uav_self_input), -1)
        # print(me_output.shape)
        # return self.policy_net, self.value_net
        return self.policy_net(me_output_pi), self.value_net(me_output_vf)

