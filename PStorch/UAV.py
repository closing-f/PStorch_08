import numpy as np
import random
from gym import spaces
from base import Agent
from itertools import zip_longest

class PointAgent(Agent):
    def __init__(self, experiment):
        super(PointAgent, self).__init__()
        self.comm_radius = experiment.comm_radius
        self.obs_radius = experiment.obs_radius
        self.obs_mode = experiment.obs_mode
        self.distance_bins = experiment.distance_bins
        self.bearing_bins = experiment.bearing_bins
        self.torus = experiment.torus
        self.n_agents = experiment.nr_agents
        self.n_evaders = experiment.nr_evaders
        self.world_size = experiment.world_size
        self.dim_a = 2
        #应该了解消耗了多少电量,默认为3
        self.energy = 3
        self.wall = 0
        self.delta_energy = 0
        self.n_step = experiment.n_step


        if self.obs_mode == 'sum_obs_no_ori':
            #self.dim_rec_o = (self.n_agents, 5)
            #这里是10是为了可以扩展到10agents的场景
            #这个可以再改写，比如就判断距离自己最近的10个agent
            self.dim_rec_o = (8, 6)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_evader_o = (self.n_evaders, 5)
            # wall and tours
            self.dim_local_o = 2
            self.dim_flat_o = np.prod(self.dim_evader_o) + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o


        self.r_matrix = None
        self.graph_feature = None
        self.see_evader = None
        self.dynamics = experiment.dynamics
        self.max_lin_velocity = 50  # cm/s
        self.max_ang_velocity = 2 * np.pi

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_pos[0] = random.randint(0, 100)
        self.state.p_pos[1] = random.randint(0, 100)
        self.state.p_orientation = state[2]
        #初始位置，速度设置为0
        self.state.p_vel = np.zeros(2)
        self.state.w_vel = np.zeros(2)
        self.graph_feature = np.inf
        self.see_evader = 0
        #能量为3
        self.energy = 3
        self.delta_energy = 0

    def is_athome(self):
        if self.state.p_pos[0] < 1100 and self.state.p_pos[1] < 1100:
            return True
        else:
            return False

    def get_observation(self, dm, my_orientation, nodes, timestep):
        evader_dists = dm[-self.n_evaders:]
        evader_bearings = my_orientation[-self.n_evaders:]
        evaders = nodes[-self.n_evaders:]
        pursuer_dists = dm[:-self.n_evaders]
        pursuer_bearings = my_orientation[:-self.n_evaders]
        pursuers = nodes[:-self.n_evaders]


        if self.obs_mode == 'sum_obs_no_ori':
            dist_to_evader = np.zeros(self.n_evaders)
            angle_to_evader = np.zeros((self.n_evaders, 2))

            sum_obs = np.zeros(self.dim_rec_o)
            pursuers_is_self = (pursuer_dists == -1.)
            # print(pursuers_is_self)
            assert any(pursuers_is_self), "Do not find uav_self"
            pursuers_in_range = (pursuer_dists < self.comm_radius) & (0 < pursuer_dists)
            num = 0
            for uav_idx, (is_self, is_neighbor) in enumerate(zip_longest(pursuers_is_self, pursuers_in_range)):
                if is_self:
                    sum_obs[-1, 0] = pursuer_dists[uav_idx] / self.comm_radius
                    sum_obs[-1, 1] = np.cos(pursuer_bearings[uav_idx])
                    sum_obs[-1, 2] = np.sin(pursuer_bearings[uav_idx])
                    sum_obs[-1, 3] = pursuers[uav_idx].energy
                    sum_obs[-1, 4] = 1
                if is_neighbor:
                    sum_obs[num, 0] = pursuer_dists[uav_idx] / self.comm_radius
                    sum_obs[num, 1] = np.cos(pursuer_bearings[uav_idx])
                    sum_obs[num, 2] = np.sin(pursuer_bearings[uav_idx])
                    #这个3貌似没有明确意义，我也忘了为啥弄个它
                    sum_obs[num, 3] = pursuers[uav_idx].energy
                    sum_obs[num, 4] = 1
                    num += 1
                sum_obs[uav_idx, 5] = 1

            local_obs = np.zeros(self.dim_local_o)
            #这里看有没要在墙的附近
            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 999):
                    self.wall = 1
                else:
                    self.wall = 0
                local_obs[0] = self.wall

            local_obs[1] = float(timestep) / self.n_step

            # local obs
            '''
            for i in range(self.n_evaders):
                if evader_dists[i] < self.obs_radius:
                    #这里的逻辑是在能够发现的evader做归一化
                    #但是因为引入了GPS，因此没有必要
                    dist_to_evader[i] = evader_dists[i] / self.obs_radius
                    angle_to_evader[i] = [np.cos(evader_bearings[i]), np.sin(evader_bearings[i])]
                else:
                    dist_to_evader[i] = 1.
                    angle_to_evader[i] = [0, 0]
            #这里直接给距离就好了
            #此时已经获得了全场的距离并且传入了所有agent的信息
            #此时已经做了稍稍的归一化
            '''
            for i in range(self.n_evaders):
                dist_to_evader[i] = evader_dists[i] / self.obs_radius
                angle_to_evader[i] = [np.cos(evader_bearings[i]), np.sin(evader_bearings[i])]

            #这里我默认了可以感知到正方形区域的所有poi
            #这里应该改写成在UAV覆盖范围内的Poi
            evader_obs = np.zeros(self.dim_evader_o)
            evader_obs[:, 0] = dist_to_evader
            evader_obs[:, 1] = angle_to_evader[:, 0]
            evader_obs[:, 2] = angle_to_evader[:, 1]
            #4是否覆盖，5覆盖时间
            for i in range(self.n_evaders):
                # 在写对单个uav的时候，这里的信息cover应该改成有几个uav同时覆盖他。
                # 另外这个120对应整个test阶段的时间步
                evader_obs[i, 3] = evaders[i].cover
                evader_obs[i, 4] = evaders[i].c_time / self.n_step

            obs = np.hstack([sum_obs.flatten(), local_obs.flatten(), evader_obs.flatten()])

        return obs

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self.position = x_2

    def set_angle(self, phi):
        assert phi.shape == (1,)
        self.angle = phi
        r_matrix_1 = np.squeeze([[np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
        r_matrix_2 = np.squeeze([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        self.r_matrix = np.dot(r_matrix_1, r_matrix_2)

    def get_local_obs_acc(self):
        local_obs = np.zeros(self.dim_local_o)
        local_obs[0] = self.state.p_vel[0] / self.max_lin_velocity
        local_obs[1] = self.state.p_vel[1] / self.max_ang_velocity

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[2] = wall_dists[closest_wall]
                local_obs[3] = np.cos(wall_angles[closest_wall])
                local_obs[4] = np.sin(wall_angles[closest_wall])
            else:
                local_obs[2] = 1
                local_obs[3:5] = 0

        return local_obs

    def get_local_obs(self):
        local_obs = np.zeros(self.dim_local_o)

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[0] = wall_dists[closest_wall]
                local_obs[1] = np.cos(wall_angles[closest_wall])
                local_obs[2] = np.sin(wall_angles[closest_wall])
            else:
                local_obs[0] = 1
                local_obs[1:3] = 0

        return local_obs

