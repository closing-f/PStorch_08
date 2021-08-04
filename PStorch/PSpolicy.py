import numpy as np
from Arguments import parse_args
from typing import Any
from stable_baselines3.common.policies import ActorCriticPolicy
from PSNetwork import PSNetwork


class PolicyNetwork(ActorCriticPolicy):
    def _forward_unimplemented(self, *input: Any) -> None:
        print(
            '???????????'
        )

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.ortho_init = False

        self.arglist = kwargs['arglist']
        del kwargs['arglist']
        super(PolicyNetwork, self).__init__(
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PSNetwork(arglist=self.arglist)


if __name__ == "__main__":
    arglist = parse_args()
    policy = PSNetwork(arglist)
    dim_rec_o = (arglist.max_nb_UAVs - 1, arglist.uav_obs_dim)
    dim_evader_o = (arglist.nb_PoIs, arglist.poi_dim)

    obs_dim_single = (arglist.max_nb_UAVs * arglist.uav_obs_dim) + 2 + (arglist.nb_PoIs * arglist.poi_dim)
    # print(obs_dim_single)
    obs_dim_overall = (arglist.nb_UAVs, obs_dim_single)
    obs_reset = np.zeros(obs_dim_overall)
    # print(obs_reset)

    for i in range(arglist.nb_UAVs):
        # print("HHA_{}".format(i))
        sum_nei_obs = np.ones(dim_rec_o) * i
        # local and time is + 2
        self_obs = np.zeros(arglist.uav_obs_dim + 2)
        sum_evader_obs = np.ones(dim_evader_o) * i
        ob_current_i = np.hstack([sum_nei_obs.flatten(), self_obs.flatten(), sum_evader_obs.flatten()])
        # print(ob_current_i)
        obs_reset[i, :] = ob_current_i

    a, b = policy.forward(obs_reset)
    print(a, b)
