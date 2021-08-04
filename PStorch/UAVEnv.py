import gym
import numpy as np
from Arguments import parse_args
from gym import spaces
from reward_form import RewardForm
from base import World
from UAV import PointAgent
from PoI import Evader
import matplotlib.pyplot as plt


class UAVnet(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, arglist):
        self.nr_agents = arglist.nb_UAVs
        self.nr_evaders = arglist.nb_PoIs
        self.comm_radius = arglist.comm_radius
        self.obs_radius = arglist.obs_radius
        self.world_size = arglist.world_size
        self.reward_form = arglist.reward_form
        # placeholder
        self.torus = arglist.torus
        self.uav_obs_dim = arglist.uav_obs_dim
        self.max_nr_agents = arglist.max_nb_UAVs
        self.poi_dim = arglist.poi_dim
        self.obs_dim_single_all = (arglist.max_nb_UAVs * arglist.uav_obs_dim) + 2 + (arglist.nb_PoIs * arglist.poi_dim)
        self.dim_a = arglist.dim_a
        self.num_envs = 1
        self.n_step = arglist.n_step
        # other parameters
        self.obs_mode = arglist.obs_mode
        self.distance_bins = arglist.distance_bins
        self.bearing_bins = arglist.bearing_bins
        self.dynamics = arglist.dynamics
        print("The reward form is {}".format(self.reward_form))
        self.RewardForm = RewardForm(arglist)
        self.world = World(arglist.world_size, arglist.torus, arglist.dynamics)
        self.world.agents = [PointAgent(self) for _ in range(self.nr_agents)]
        [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]
        self.ax = None

    @property
    def agents(self):
        return self.world.policy_agents

    @property
    def observation_space(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self.obs_dim_single_all,), dtype=np.float32)
        return ob_space

    @property
    def action_space(self):
        return spaces.Box(low=0., high=+1., shape=(self.dim_a,), dtype=np.float32)

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        if self.RewardForm.timestep >= self.timestep_limit:
            return True
        return False

    def reset(self):
        """
        dim_rec_o = (self.max_nr_agents - 1, self.uav_obs_dim)
        dim_evader_o = (self.nr_evaders, self.poi_dim)
        obs_reset = []

        for i in range(self.nr_agents):
            sum_nei_obs = np.ones(dim_rec_o)
            # local and time is + 2
            self_obs = np.zeros(self.uav_obs_dim + 2)
            sum_evader_obs = np.ones(dim_evader_o)
            ob_current_i = np.hstack([sum_nei_obs.flatten(), self_obs.flatten(), sum_evader_obs.flatten()])
            obs_reset.append(ob_current_i)
        # print(obs_reset)
        obs_reset = np.array(obs_reset)

        :return: obs_reset -> list
        """
        self.RewardForm.reset()
        self.world.agents = [PointAgent(self) for _ in range(self.nr_agents)]
        [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]
        pursuers = np.zeros((self.nr_agents, 3))
        evader = np.zeros((100, 2))
        for i in range(10):
            for j in range(10):
                evader[int(i * 10 + j)][0] = 50 + i * 100
                evader[int(i * 10 + j)][1] = 50 + j * 100
        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()

        obs_reset = []
        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.agents,
                                     self.RewardForm.timestep
                                     )
            obs_reset.append(ob)

        # print("obs_reset: {}".format(obs_reset))
        return obs_reset

    def step(self, actions=None):
        """
        dim_rec_o = (self.max_nr_agents - 1, self.uav_obs_dim)
        dim_evader_o = (self.nr_evaders, self.poi_dim)
        new_obs = []
        infos = []
        for i in range(self.nr_agents):
            sum_nei_obs = np.ones(dim_rec_o)
            # local and time is + 2
            self_obs = np.zeros(self.uav_obs_dim + 2)
            sum_evader_obs = np.ones(dim_evader_o)
            ob_current_i = np.hstack([sum_nei_obs.flatten(), self_obs.flatten(), sum_evader_obs.flatten()])
            new_obs.append(ob_current_i)
        # print(new_obs)
        rewards = np.zeros(self.nr_agents, dtype=np.float32)
        # print(rewards)
        dones = np.zeros(self.nr_agents)
        new_obs = np.array(new_obs)

        :param actions: UAV_actions -> self.nr_agents * dim_a = self.nr_agents * 2
        :return: next_obs, r, dones -> 1 is terminal, info -> List[Dict[str, Any]]
        """
        self.RewardForm.timestep += 1
        assert len(actions) == self.nr_agents
        # print(actions.shape)
        for agent, action in zip(self.agents, actions):
            action = action.flatten()
            action[1] = action[1] * 2 - 1
            # print("action_1： {}".format(action[1]))
            agent.action.u = action[0:2]
            # if self.world.dim_c > 0:
            #     agent.action.c = action[2:]
        # 距离原点的距离
        distance_now = self.world.step()
        next_obs = []
        dones = np.zeros(self.nr_agents)
        for i, bot in enumerate(self.world.policy_agents):
            # print(hop_counts)
            # bot_in_subset = [list(s) for s in sets if i in s]
            # [bis.remove(i) for bis in bot_in_subset]
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.agents,
                                     self.RewardForm.timestep)
            next_obs.append(ob)
            dones[i] = self.is_terminal
        if self.reward_form == 'LT':
            r = self.RewardForm.LT(self.world.agents)
        if self.reward_form == 'PC':
            r = self.RewardForm.PC(self.world.agents)
        if self.reward_form == 'IA':
            r = self.RewardForm.IA(self.world.agents)
        if self.reward_form == 'RE':
            r = self.RewardForm.RE(self.world.agents)
        if self.reward_form == 'AL':
            r = self.RewardForm.AL(self.world.agents)


        info = [{'pursuer_states': self.world.agent_states,
                'evader_states': self.world.landmark_states,
                'actions': actions,
                'distance': distance_now}]

        if dones[0] == True:
            print("The env is reset")
            self.reset()

        return next_obs, r, dones, info

    def render(self, mode='human'):
        if not self.ax:
            fig, ax = plt.subplots()
            self.ax = ax
        else:
            self.ax.clear()

        self.ax.set_aspect('equal')
        self.ax.set_xlim((0, self.world_size))
        self.ax.set_ylim((0, self.world_size))

        comm_circles = []
        obs_circles = []
        self.ax.scatter(self.world.landmark_states[:, 0], self.world.landmark_states[:, 1], c='r', s=20)
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=20)

        for i in range(self.nr_agents):
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.comm_radius, color='g', fill=False))
            self.ax.add_artist(comm_circles[i])

            obs_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                           self.world.agent_states[i, 1]),
                                          self.obs_radius, color='g', fill=False))
            self.ax.add_artist(obs_circles[i])

            # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
            #              "{}".format(i), ha='center',
            #              va='center', size=20)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])
        plt.pause(0.01)

    def close(self):
        pass

if __name__ == "__main__":
    arglist = parse_args()
    env = UAVnet(arglist)
    env.step()
    print(env.action_space)
    print(env.observation_space)
