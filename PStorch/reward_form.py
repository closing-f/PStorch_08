import numpy as np

class RewardForm(object):
    def __init__(self, arglist):
        self.nr_agents = arglist.nb_UAVs
        self.nr_evaders = arglist.nb_PoIs
        self.c_time_last = np.zeros((self.nr_agents,))
        self.c_time_last_total = 0
        self.energy_last = 3 * np.ones((self.nr_agents,))
        self.energy_last_total = 3 * self.nr_agents
        self.total_reward = np.zeros((self.nr_agents,))
        self.low_power_loss_flag = np.ones((self.nr_agents,))
        self.timestep = None
        self.n_step = 1024
        self.beta = arglist.beta
        self.alpha = arglist.beta

    def reset(self):
        self.timestep = 0
        self.c_time_last = np.zeros((self.nr_evaders,))
        self.c_time_last_total = 0
        self.total_reward = np.zeros((self.nr_agents,))
        self.low_power_loss_flag = np.ones((self.nr_agents,))
        self.energy_last = 3 * np.ones((self.nr_agents,))
        self.energy_last_total = 3 * self.nr_agents

    def LT(self, nodes):
        timesteps_per_batch = self.n_step  # type: int
        evaders = nodes[-self.nr_evaders:]
        pursuers = nodes[:-self.nr_evaders]
        f_up = 0
        f_down = 0

        # 能量消耗 （shape ：nr_agent) ，初值即为上次的剩余电量
        delta_energy = np.zeros((self.nr_agents,))
        # ctime部分奖励 （shape ：nr_agent)，初值为零
        ctime_reward = np.zeros((self.nr_agents,))

        wall_sub = 15  # 碰墙对于每个agent的惩罚数值
        # 综合reward

        reward = np.zeros((self.nr_agents,))

        for i in range(self.nr_evaders):
            # 一个 pol 可供分ctime的 reward ,还要除一个时间步数（归一化）
            pol_delta_ctime = float(evaders[i].c_time - self.c_time_last[i]) / timesteps_per_batch
            # 一个 pol 需要分的ctime的 数目
            cover_len = len(evaders[i].sub_list)
            if cover_len > 0:
                for j in evaders[i].sub_list:
                    # 每一个agent的奖励 = 每一个pol能给出的总数 / agent覆盖个数
                    ctime_reward[j] += float(pol_delta_ctime) / cover_len

            f_up += evaders[i].c_time
            f_down += np.square(float(evaders[i].c_time) / timesteps_per_batch)

            # 更新c_time_last
            self.c_time_last[i] = evaders[i].c_time

        # print("pol finish f_up:",f_up," f_down:",f_down)

        for i in range(self.nr_agents):

            # 每一个agent的delta能量计算、最新能量更新
            delta_energy[i] = self.energy_last[i] - pursuers[i].energy

            self.energy_last[i] = pursuers[i].energy

            # 如果有agent碰边界，自己的reward - 1
            if pursuers[i].wall == 1:
                reward[i] -= wall_sub

        # f的计算，这里没有变，self.c_time_last改成了self.c_time_last_total，原名称为矢量，total为标量
        self.c_time_last_total = f_up
        f_up = np.square(float(f_up) / timesteps_per_batch)
        f_down *= 100

        #    print("agent finish  f_down:", f_down," energy:",self.energy_last.mean()," delta_energy:",delta_energy.mean())

        if f_down == 0:
            print("error：f_down is zero ,f_up:", f_up)
        f = f_up / f_down

        for i in range(self.nr_agents):

            # 这里只需要加上 ctime奖励在单位能量中，带f权重的奖励，关于wall惩罚已经扣掉了
            if self.energy_last[i] > 0.00001:
                # print("step:",self.timestep," i:",i," wall:",reward[i]," add:",( f * ctime_reward[i] / delta_energy[i]))
                reward[i] += f * ctime_reward[i] / delta_energy[i]
            else:
                if self.low_power_loss_flag[i] == 1:
                    # print("low power loss of ", i)
                    # 中途没电惩罚：
                    self.low_power_loss_flag[i] = 0
                reward[i] -= 1

            # 结束回家惩罚判断
            if self.timestep == timesteps_per_batch - 2:
                #    print("total reward:",self.total_reward)
                if pursuers[i].is_athome() is False:
                    reward[i] -= (pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5


                    # print("Not at home loss of ", i, ",loss:",
                    #       ((pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5))
        # print("step:", self.timestep, " total reward:", self.total_reward)
        # print(reward)
        return reward

    def PC(self, nodes):
        timesteps_per_batch = self.n_step  # type: int
        evaders = nodes[-self.nr_evaders:]
        pursuers = nodes[:-self.nr_evaders]
        f_up = 0
        f_down = 0
        e_now = 0
        walls = 0

        for i in range(self.nr_evaders):
            f_up += evaders[i].c_time
            f_down += np.square(float(evaders[i].c_time) / timesteps_per_batch)
        for i in range(self.nr_agents):

            e_now += pursuers[i].energy

            if pursuers[i].wall == 1:
                walls += 1

        delta_c_time = float(f_up - self.c_time_last_total) / timesteps_per_batch
        delta_energy = self.energy_last_total - e_now


        self.c_time_last_total = f_up
        self.energy_last_total = e_now

        f_up = np.square(float(f_up) / timesteps_per_batch)
        f_down = f_down * 100
        if f_down == 0:
            print("error")
        f = f_up / f_down

        if delta_energy < 0.002 * self.nr_agents:
            delta_energy = 0.002 * self.nr_agents

        r = f * delta_c_time / delta_energy
        # 这里还没有写loss的部分

        if r == float("inf"):
            # 对应不变化的情况
            r = 0

        # print("step:",self.timestep,"r:",r)
        r = np.ones((self.nr_agents,)) * r

        for i in range(self.nr_agents):
            # 撞墙惩罚写入
            if pursuers[i].wall == 1:
                r[i] -= 15
            # 结束回家惩罚判断
            if self.timestep == timesteps_per_batch - 2:
                #    print("total reward:",self.total_reward)
                if pursuers[i].is_athome() is False:
                    r[i] -= ((pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5)
                #    print("Not at home loss of ", i)

        return r

    def IA(self, nodes):
        timesteps_per_batch = self.n_step  # type: int
        evaders = nodes[-self.nr_evaders:]
        pursuers = nodes[:-self.nr_evaders]
        f_up = 0
        f_down = 0

        # 能量消耗 （shape ：nr_agent) ，初值即为上次的剩余电量
        delta_energy = np.zeros((self.nr_agents,))
        # ctime部分奖励 （shape ：nr_agent)，初值为零
        ctime_reward = np.zeros((self.nr_agents,))

        wall_sub = 15  # 碰墙对于每个agent的惩罚数值
        # 综合reward

        reward = np.zeros((self.nr_agents,))

        for i in range(self.nr_evaders):
            # 一个 pol 可供分ctime的 reward ,还要除一个时间步数（归一化）
            pol_delta_ctime = float(evaders[i].c_time - self.c_time_last[i]) / timesteps_per_batch
            # 一个 pol 需要分的ctime的 数目
            cover_len = len(evaders[i].sub_list)
            if cover_len > 0:
                for j in evaders[i].sub_list:
                    # 每一个agent的奖励 = 每一个pol能给出的总数 / agent覆盖个数
                    ctime_reward[j] += float(pol_delta_ctime) / cover_len

            f_up += evaders[i].c_time
            f_down += np.square(float(evaders[i].c_time) / timesteps_per_batch)

            # 更新c_time_last
            self.c_time_last[i] = evaders[i].c_time

        # print("pol finish f_up:",f_up," f_down:",f_down)

        for i in range(self.nr_agents):

            # 每一个agent的delta能量计算、最新能量更新
            delta_energy[i] = self.energy_last[i] - pursuers[i].energy

            self.energy_last[i] = pursuers[i].energy

            # 如果有agent碰边界，自己的reward - 1
            if pursuers[i].wall == 1:
                reward[i] -= wall_sub

        # f的计算，这里没有变，self.c_time_last改成了self.c_time_last_total，原名称为矢量，total为标量
        self.c_time_last_total = f_up
        f_up = np.square(float(f_up) / timesteps_per_batch)
        f_down *= 100

        #    print("agent finish  f_down:", f_down," energy:",self.energy_last.mean()," delta_energy:",delta_energy.mean())

        if f_down == 0:
            print("error：f_down is zero ,f_up:", f_up)
        f = f_up / f_down

        for i in range(self.nr_agents):

            # 这里只需要加上 ctime奖励在单位能量中，带f权重的奖励，关于wall惩罚已经扣掉了
            if self.energy_last[i] > 0.00001:
                # print("step:",self.timestep," i:",i," wall:",reward[i]," add:",( f * ctime_reward[i] / delta_energy[i]))
                reward[i] += f * ctime_reward[i] / delta_energy[i]
            else:
                if self.low_power_loss_flag[i] == 1:
                    # print("low power loss of ", i)
                    # 中途没电惩罚：
                    self.low_power_loss_flag[i] = 0
                reward[i] -= 1

            # 结束回家惩罚判断
            if self.timestep == timesteps_per_batch - 2:
                # print("total reward:",self.total_reward)
                # print(pursuers[i].is_athome)
                if pursuers[i].is_athome() is False:
                    reward[i] -= (pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5
                    # print("Not at home loss of ", i, ",loss:",
                    #      ((pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5))

        # 情感惩罚机制
        self.total_reward += reward
        # print("step:",self.timestep,self.total_reward)
        emotion_alpha = self.alpha / ((self.nr_agents - 1) * timesteps_per_batch)
        emotion_beta = self.beta / ((self.nr_agents - 1) * timesteps_per_batch)

        for i in range(self.nr_agents):
            max_reward = max(self.total_reward)
            if max_reward > self.total_reward[i]:
                # 存在嫉妒
                #   print("嫉妒 ", i,emotion_alpha * (max_reward - self.total_reward[i]) )
                reward[i] -= emotion_beta * (max_reward - self.total_reward[i])
            min_reward = min(self.total_reward)
            if min_reward < self.total_reward[i]:
                # 存在骄傲
                #   print("骄傲 ", i,emotion_beta * (self.total_reward[i] - min_reward) )
                reward[i] -= emotion_alpha * (self.total_reward[i] - min_reward)

        # print("step:",self.timestep," total reward:", self.total_reward)

        return reward

    def RE(self, nodes):
        timesteps_per_batch = self.n_step  # type: int
        evaders = nodes[-self.nr_evaders:]
        pursuers = nodes[:-self.nr_evaders]
        f_up = 0
        f_down = 0

        # 能量消耗 （shape ：nr_agent) ，初值即为上次的剩余电量
        delta_energy = np.zeros((self.nr_agents,))
        # ctime部分奖励 （shape ：nr_agent)，初值为零
        ctime_reward = np.zeros((self.nr_agents,))

        wall_sub = 15  # 碰墙对于每个agent的惩罚数值
        # 综合reward

        reward = np.zeros((self.nr_agents,))

        for i in range(self.nr_evaders):
            # 一个 pol 可供分ctime的 reward ,还要除一个时间步数（归一化）
            pol_delta_ctime = float(evaders[i].c_time - self.c_time_last[i]) / timesteps_per_batch
            # 一个 pol 需要分的ctime的 数目
            cover_len = len(evaders[i].sub_list)
            if cover_len > 0:
                for j in evaders[i].sub_list:
                    # 每一个agent的奖励 = 每一个pol能给出的总数 / agent覆盖个数
                    ctime_reward[j] += float(pol_delta_ctime) / cover_len

            f_up += evaders[i].c_time
            f_down += np.square(float(evaders[i].c_time) / timesteps_per_batch)

            # 更新c_time_last
            self.c_time_last[i] = evaders[i].c_time

            # exploit 奖励
            if len(evaders[i].exploiter) > 0:
                x = evaders[i].state.p_pos[0] / 100 + 0.5
                y = evaders[i].state.p_pos[1] / 100 + 0.5
                # print("exploiter ", evaders[i].exploiter[0] ," x:",x," y:",y,
                #   " reward:", ((x**2+y**2)**0.5) * 3 * (timesteps_per_batch - self.timestep) / timesteps_per_batch)
                for i in evaders[i].exploiter:
                    reward[i] += ((x ** 2 + y ** 2) ** 0.5) * 3 * (
                                timesteps_per_batch - self.timestep) / timesteps_per_batch
        # print("pol finish f_up:",f_up," f_down:",f_down)

        for i in range(self.nr_agents):

            # 每一个agent的delta能量计算、最新能量更新
            delta_energy[i] = self.energy_last[i] - pursuers[i].energy

            self.energy_last[i] = pursuers[i].energy

            # 如果有agent碰边界，自己的reward - 1
            if pursuers[i].wall == 1:
                reward[i] -= wall_sub

        # f的计算，这里没有变，self.c_time_last改成了self.c_time_last_total，原名称为矢量，total为标量
        self.c_time_last_total = f_up
        f_up = np.square(float(f_up) / timesteps_per_batch)
        f_down *= 100

        # print("agent finish  f_down:", f_down," energy:",self.energy_last.mean()," delta_energy:",delta_energy.mean())

        if f_down == 0:
            print("error：f_down is zero ,f_up:", f_up)
        f = f_up / f_down

        for i in range(self.nr_agents):

            # 这里只需要加上 ctime奖励在单位能量中，带f权重的奖励，关于wall惩罚已经扣掉了
            if self.energy_last[i] > 0.00001:
                # print("step:",self.timestep," i:",i," wall:",reward[i]," add:",( f * ctime_reward[i] / delta_energy[i]))
                reward[i] += f * ctime_reward[i] / delta_energy[i]
            else:
                if self.low_power_loss_flag[i] == 1:
                    # print("low power loss of ", i)
                    # 中途没电惩罚：
                    self.low_power_loss_flag[i] = 0
                reward[i] -= 1

            # 结束回家惩罚判断
            if self.timestep == timesteps_per_batch - 2:
                #    print("total reward:",self.total_reward)
                if pursuers[i].is_athome() is False:
                    reward[i] -= (pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5


        return reward

    def AL(self, nodes):
        timesteps_per_batch = self.n_step  # type: int
        evaders = nodes[-self.nr_evaders:]
        pursuers = nodes[:-self.nr_evaders]
        f_up = 0
        f_down = 0

        # 能量消耗 （shape ：nr_agent) ，初值即为上次的剩余电量
        delta_energy = np.zeros((self.nr_agents,))
        # ctime部分奖励 （shape ：nr_agent)，初值为零
        ctime_reward = np.zeros((self.nr_agents,))

        wall_sub = 15  # 碰墙对于每个agent的惩罚数值
        # 综合reward

        reward = np.zeros((self.nr_agents,))

        for i in range(self.nr_evaders):
            # 一个 pol 可供分ctime的 reward ,还要除一个时间步数（归一化）
            pol_delta_ctime = float(evaders[i].c_time - self.c_time_last[i]) / timesteps_per_batch
            # 一个 pol 需要分的ctime的 数目
            cover_len = len(evaders[i].sub_list)
            if cover_len > 0:
                for j in evaders[i].sub_list:
                    # 每一个agent的奖励 = 每一个pol能给出的总数 / agent覆盖个数
                    ctime_reward[j] += float(pol_delta_ctime) / cover_len

            f_up += evaders[i].c_time
            f_down += np.square(float(evaders[i].c_time) / timesteps_per_batch)

            # 更新c_time_last
            self.c_time_last[i] = evaders[i].c_time

            # exploit 奖励
            if len(evaders[i].exploiter) > 0:
                x = evaders[i].state.p_pos[0] / 100 + 0.5
                y = evaders[i].state.p_pos[1] / 100 + 0.5
                # print("exploiter ", evaders[i].exploiter[0] ," x:",x," y:",y,
                # " reward:", ((x**2+y**2)**0.5) * 3 * (timesteps_per_batch - self.timestep) / timesteps_per_batch)
                for i in evaders[i].exploiter:
                    reward[i] += ((x ** 2 + y ** 2) ** 0.5) * 3 * (
                                timesteps_per_batch - self.timestep) / timesteps_per_batch
        # print("pol finish f_up:",f_up," f_down:",f_down)

        for i in range(self.nr_agents):

            # 每一个agent的delta能量计算、最新能量更新
            delta_energy[i] = self.energy_last[i] - pursuers[i].energy

            self.energy_last[i] = pursuers[i].energy

            # 如果有agent碰边界，自己的reward - 1
            if pursuers[i].wall == 1:
                reward[i] -= wall_sub

        # f的计算，这里没有变，self.c_time_last改成了self.c_time_last_total，原名称为矢量，total为标量
        self.c_time_last_total = f_up
        f_up = np.square(float(f_up) / timesteps_per_batch)
        f_down *= 100

        #print("agent finish  f_down:", f_down," energy:",self.energy_last.mean(),
        # " delta_energy:",delta_energy.mean())

        if f_down == 0:
            print("error：f_down is zero ,f_up:", f_up)
        f = f_up / f_down

        for i in range(self.nr_agents):

            # 这里只需要加上 ctime奖励在单位能量中，带f权重的奖励，关于wall惩罚已经扣掉了
            if self.energy_last[i] > 0.0002:
                # print("step:",self.timestep," i:",i," wall:",
                # reward[i]," add:",( f * ctime_reward[i] / delta_energy[i]))
                reward[i] += f * ctime_reward[i] / delta_energy[i]
            else:
                if self.low_power_loss_flag[i] == 1:
                    # print("low power loss of ", i)
                    # 中途没电惩罚：
                    self.low_power_loss_flag[i] = 0
                reward[i] -= 1

            # 结束回家惩罚判断
            if self.timestep == timesteps_per_batch - 2:
                #    print("total reward:",self.total_reward)
                if pursuers[i].is_athome() is False:
                    reward[i] -= (pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5
                    # print("Not at home loss of ", i, ",loss:",
                    #       ((pursuers[i].state.p_pos[0] ** 2 + pursuers[i].state.p_pos[1] ** 2) ** 0.5))

        # 情感惩罚机制
        self.total_reward += reward
        # print("step:",self.timestep,self.total_reward)
        emotion_alpha = self.alpha / ((self.nr_agents - 1) * timesteps_per_batch)
        emotion_beta = self.beta / ((self.nr_agents - 1) * timesteps_per_batch)

        for i in range(self.nr_agents):
            max_reward = max(self.total_reward)
            if max_reward > self.total_reward[i]:
                # 存在嫉妒
                #       print("嫉妒 ", i,emotion_alpha * (max_reward - self.total_reward[i]) )
                reward[i] -= emotion_beta * (max_reward - self.total_reward[i])
            min_reward = min(self.total_reward)
            if min_reward < self.total_reward[i]:
                # 存在骄傲
                #       print("骄傲 ", i,emotion_beta * (self.total_reward[i] - min_reward) )
                reward[i] -= emotion_alpha * (self.total_reward[i] - min_reward)

        return reward

