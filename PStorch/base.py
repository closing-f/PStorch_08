import numpy as np
import utils as U



dynamics = ['point', 'unicycle', 'box2d', 'direct', 'unicycle_acc']


class EntityState(object):
    """physical/external base state of all entities"""
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_orientation = None
        # physical velocity
        self.p_vel = None
        # velocity in world coordinates
        self.w_vel = None


class AgentState(EntityState):
    """state of agents (including communication and internal/mental state)"""
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


class Action(object):
    """action of the agent"""
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity(object):
    """properties and state of physical world entity"""
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):
    """properties of landmark entities"""
    def __init__(self):
        super(Landmark, self).__init__()


class TransportSource(Landmark):
    def __init__(self, nr_items):
        super(TransportSource, self).__init__()
        self.nr_items = nr_items

    def reset(self, state):
        self.state.p_pos = state[0:2]


class TransportSink(Landmark):
    def __init__(self):
        super(TransportSink, self).__init__()
        self.nr_items = 0

    def reset(self, state):
        self.state.p_pos = state[0:2]


class Agent(Entity):
    """properties of agent entities"""
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # physical damping
        self.lin_damping = 0.01  # 0.025  # 0.05
        self.ang_damping = 0.01  # 0.05
        self.max_lin_velocity = 50  # cm/s
        self.max_ang_velocity = np.pi  # 2 * np.pi  # rad/s
        self.max_lin_acceleration = 10  # 25  # 100  # cm/s**2
        self.max_ang_acceleration = np.pi  # 60  # rad/s**2


class World(object):
    def __init__(self, world_size, torus, agent_dynamic):
        self.nr_agents = None
        # world is square
        self.world_size = world_size
        # dynamics of agents
        assert agent_dynamic in dynamics
        self.agent_dynamic = agent_dynamic
        # periodic or closed world
        self.torus = torus
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # matrix containing agent states
        self.agent_states = None
        # matrix containing landmark states
        self.landmark_states = None
        # x,y of everything
        self.nodes = None
        self.distance_matrix = None
        self.angle_matrix = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.01
        self.action_repeat = 10
        self.timestep = 0
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3


    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def reset(self):
        self.timestep = 0
        self.nr_agents = len(self.policy_agents)

        for i, agent in enumerate(self.policy_agents):
            agent.reset(self.agent_states[i, :])

        for i, agent in enumerate(self.scripted_agents):
            agent.reset(self.landmark_states[i, :])


        self.nodes = np.vstack(
            [self.agent_states[:, 0:2], self.landmark_states]) if self.landmark_states is not None else self.agent_states[:,
                                                                                                 0:2]

        #这里的矩阵都是用来计算uav与poi之间的相对距离的。
        self.distance_matrix = U.get_distance_matrix(self.nodes,
                                                     torus=self.torus, world_size=self.world_size, add_to_diagonal=-1)

        angles = np.vstack([U.get_angle(self.nodes, a[0:2],
                                        torus=self.torus, world_size=self.world_size) - a[2] for a in self.agent_states])
        angles_shift = -angles % (2 * np.pi)
        self.angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)


        for i, agent in enumerate(self.scripted_agents):
            agent.exploit_init(self)

    def step(self):

        self.timestep += 1

        for i, agent in enumerate(self.scripted_agents):
            action = agent.action_callback(self)
            #这里Poi的位置改变恒为0~，所以可以忽略下面的这一段
            #但是这个act_callback是要执行的，因为该函数里要改变c——time等因素
            if agent.dynamics == 'direct':
                next_coord = agent.state.p_pos + action * agent.max_speed * self.dt * self.action_repeat
                if self.torus:
                    next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                    next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
                else:
                    next_coord = np.where(next_coord < 0, 0, next_coord)
                    next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)
                agent.state.p_pos = next_coord

            self.landmark_states[i, :] = agent.state.p_pos


        if self.agent_dynamic == 'unicycle':
            # unicycle dynamics
            scaled_actions = np.zeros([self.nr_agents, 2])

            for i, agent in enumerate(self.policy_agents):
                scaled_actions[i, 0] = agent.action.u[0] * agent.max_lin_velocity
                scaled_actions[i, 1] = agent.action.u[1] * agent.max_ang_velocity
                # 这里来改变能量
                # 当能量小于0的时候，就不允许动了
                # 我这里没有写其他的功能，但是可以在这里实现一下，比如自己飞回来之类的
                if agent.energy < 0.00001:
                    scaled_actions[i, 0] = 0
                    agent.energy = 0
                else:
                    # 开始消耗能量~
                    if abs(scaled_actions[i, 0]) >= 0:
                        # 在我们算能量的时候，动作应当从1变成在0.5
                        v = agent.action.u[0] * 0.5
                        delta_E = 1/3.0 * v ** 3 - 0.0625 * v + 0.03
                        delta_E = delta_E / 15
                        delta_E += 0.001

                        agent.energy = agent.energy - delta_E
                    if agent.energy < 0.00001:
                        agent.energy = 0

            #这里主要是让仿真在角度改变上更精准，尤其是处理边界的时候
            for i in range(self.action_repeat):
                step = np.concatenate([scaled_actions[:, [0]] * np.cos(self.agent_states[:, 2:3]),
                                       scaled_actions[:, [0]] * np.sin(self.agent_states[:, 2:3])],
                                      axis=1)
                next_coord = self.agent_states[:, 0:2] + step * self.dt
                next_angle = (self.agent_states[:, 2:3] + scaled_actions[:, [1]] * self.dt) % (2 * np.pi)

                if self.torus:
                    next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                    next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
                else:
                    next_coord = np.where(next_coord < 0, 0, next_coord)
                    next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)

                agent_states_next = np.concatenate([next_coord, next_angle], axis=1)

                self.agent_states = agent_states_next

            distance_now = np.zeros(self.nr_agents)
            for i, agent in enumerate(self.policy_agents):
                agent.state.p_pos = agent_states_next[i, 0:2]
                agent.state.p_orientation = agent_states_next[i, 2:3]
                agent.state.p_vel = step[i, :]
                distance_now[i] = agent.state.p_pos[0]**2 + agent.state.p_pos[1]**2



        self.nodes = np.vstack(
            [self.agent_states[:, 0:2], self.landmark_states]) if self.landmark_states is not None else self.agent_states[:, 0:2]

        self.distance_matrix = U.get_distance_matrix(self.nodes,
                                                     torus=self.torus, world_size=self.world_size, add_to_diagonal=-1)

        angles = np.vstack([U.get_angle(self.nodes, a[0:2],
                                        torus=self.torus, world_size=self.world_size) - a[2] for a in
                            self.agent_states])
        angles_shift = -angles % (2 * np.pi)
        self.angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)

        return distance_now