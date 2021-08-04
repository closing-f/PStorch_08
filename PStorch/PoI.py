import numpy as np
import utils as U
from base import Agent


class Evader(Agent):
    def __init__(self, experiment):
        super(Evader, self).__init__()
        self.obs_radius = experiment.obs_radius
        self.world_size = experiment.world_size
        self.torus = experiment.torus
        self.dynamics = 'direct'
        self.bounding_box = np.array([0., self.world_size, 0., self.world_size])
        self.max_speed = 2 * 10  # cm/s
        self.action_callback = self.step

        self.exploit_flag = 1
        self.exploiter = []
        #表明是否被覆盖,默认未被覆盖
        self.cover = 0
        #应该让他知道被覆盖了多长的时间
        self.c_time = 0
        self.covered = 0
        self.sub_list = []

    def reset(self, state):
        self.state.p_pos = state
        self.state.p_vel = np.zeros(2)
        self.c_time = 0
        self.cover = 0
        self.covered = 0
        self.exploit_flag = 1
        self.sub_list = []
        self.exploiter = []

    def exploit_init(self, world):
        nodes = np.vstack([world.agent_states[:, 0:2],
                           self.state.p_pos,
                           ])
        distances = U.get_euclid_distances(nodes)
        evader_dist = distances[-1, :-1]
        closest_pursuer = np.where(evader_dist == evader_dist.min())[0]
        self.sub_list = list(np.where(evader_dist < self.obs_radius)[0])

        if len(self.sub_list) > 0:
        #    print("exploit init:",self.state.p_pos)
            self.exploit_flag = 0

    def step(self, world):

        # exploiter 初始化
        self.exploiter = []

        nodes = np.vstack([world.agent_states[:, 0:2],
                           self.state.p_pos,
                           ])
        distances = U.get_euclid_distances(nodes)
        evader_dist = distances[-1, :-1]
        closest_pursuer = np.where(evader_dist == evader_dist.min())[0]
        self.sub_list = list(np.where(evader_dist < self.obs_radius)[0])

        energy_cover = False
        if len(self.sub_list) > 0:
            for i in self.sub_list:
                if world.policy_agents[i].energy > 0:
                    energy_cover = True

            # exploiter记录
            if self.exploit_flag == 1:
                self.exploit_flag = 0
                self.exploiter = self.sub_list

        if energy_cover:
            self.cover = 1
            self.c_time += 1
            self.covered = 1
        else:
            self.cover = 0

        if len(self.sub_list) > 10:
            self.sub_list = list(np.argsort(evader_dist)[0:10])
        #    self.sub_list.append(world.nr_agents)


        d = np.zeros(2)

        return d
