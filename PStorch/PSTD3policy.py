from abc import ABC
from stable_baselines3.td3.policies import TD3Policy
from typing import Optional
from PSActorCritic import PSActor, PSCritic
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)


class PSTD3policy(TD3Policy, ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.ortho_init = False

        super(PSTD3policy, self).__init__(
            *args,
            **kwargs
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PSActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return PSActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PSCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PSCritic(**critic_kwargs).to(self.device)




