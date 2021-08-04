import copy
import math
import os
import random
import torch as th
import numpy as np


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

def eval_multiUAV(arglist,
                  uav_env,
                  model,
                  ModelNumber,
                  if_render=False,
                  path=None):
    setup_seed(arglist.seed)
    episodes = arglist.episodes
    render_eval = if_render
    reward_form = arglist.reward_form

    env = uav_env
    nr_agent = env.nr_agents
    nr_evaders = env.nr_evaders
    nb_eval_steps = 1024

    num_energy_zero = 0
    num_at_home = 0
    dis_arr = np.zeros([episodes, nr_agent], dtype=np.float)
    step_arr = np.zeros([episodes, nr_agent], dtype=np.int)
    final_dis = np.zeros([episodes, nr_agent], dtype=np.float)
    max_distance = np.zeros(episodes)
    ACS = np.zeros(episodes)
    FI = np.zeros(episodes)
    NAEC = np.zeros(episodes)

    for ep in range(episodes):
        print(ep)
        obs = env.reset()
        nodes = env.world.agents
        evaders = nodes[-env.nr_evaders:]
        pursuers = nodes[:-env.nr_evaders]
        energy = 3 * np.ones(nr_agent)
        c_time = np.zeros(nr_evaders)
        c_time_square = np.zeros(nr_evaders)
        if render_eval:
            env.render()

        for t in range(nb_eval_steps):
            ac, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(ac)
            if render_eval:
                env.render()

            distance = info[0]['distance']
            for idx in range(nr_agent):
                if distance[idx] > dis_arr[ep, idx] and t < 300:
                    dis_arr[ep, idx] = distance[idx]
                    step_arr[ep, idx] = t
            #   print(distance)
            tmp_max_distance = max(distance)
            #   print(tmp_max_distance)
            if tmp_max_distance > max_distance[ep]:
                max_distance[ep] = tmp_max_distance
            if t == nb_eval_steps - 2:
                for idx in range(nr_agent):
                    final_dis[ep, idx] = distance[idx]**0.5
                    print(final_dis[ep, idx])

        for poi_idx in range(env.nr_evaders):
            c_time[poi_idx] = copy.deepcopy(evaders[poi_idx].c_time)
            c_time_square[poi_idx] = np.square(c_time[poi_idx]/nb_eval_steps)

        for idx in range(env.nr_agents):
            energy[idx] = copy.deepcopy(pursuers[idx].energy)
            if pursuers[idx].is_athome() is True:
                num_at_home += 1
            if energy[idx] < 0.000001:
                print("agent ", idx," low power",)
                num_energy_zero += 1

        c_time = c_time / nb_eval_steps
        SumCTime = sum(c_time)
        SumCTimeSquare = sum(c_time_square)
        #这里的100是PoI的个数
        AverageCoverageScore = SumCTime / nr_evaders
        f_up = np.square(SumCTime)
        f_down = nr_evaders * SumCTimeSquare
        FairnessIndex = f_up/f_down
        NormalizedAverageEnergyC = 1 - sum(energy / 3) / nr_agent
        ACS[ep] = AverageCoverageScore
        FI[ep] = FairnessIndex
        NAEC[ep] = NormalizedAverageEnergyC

    acs = sum(ACS) / episodes
    fi = sum(FI) / episodes
    naec = sum(NAEC) / episodes
    ee = acs * fi / naec
    ave_num_energy_zero = num_energy_zero / episodes
    ave_num_at_home = num_at_home / episodes
    ave_max_distance = math.sqrt(sum(max_distance) / episodes)
    ave_final_distance = final_dis.mean()
    min_final_distance = final_dis.min(axis=1).mean()
    max_final_distance = final_dis.max(axis=1).mean()
    # info_epsiodes = np.array([arglist.LoadNumber, acs, fi, naec, ee])
    print('integer:{}\t'.format(ModelNumber))
    print('{}\t'.format(acs))  # average coverage score
    print('{}\t'.format(fi))   # fairness index
    print('{}\t'.format(naec)) # normalized average energy C
    print('{}\t'.format(ee))
    print('{}\t'.format(ave_num_energy_zero))
    print('{}\t'.format(ave_max_distance))
    print('{}\t'.format(ave_final_distance))
    print('{}\t'.format(min_final_distance))
    print('{}\n'.format(max_final_distance))
    print('{}\t'.format(ave_num_at_home))
    print(step_arr)
    print(dis_arr)
    print(final_dis)
    print(ave_max_distance)

    file_handle = open(os.path.join(path, '{}.txt').format(reward_form), mode='a')
    file_handle.write('{}\t'.format(ModelNumber))
    file_handle.write('{}\t'.format(nr_agent))
    file_handle.write('{:.8f}\t'.format(acs))
    file_handle.write('{:.8f}\t'.format(fi))
    file_handle.write('{:.8f}\t'.format(naec))
    file_handle.write('{:.8f}\t'.format(ee))
    file_handle.write('{:.4f}\t'.format(ave_num_energy_zero))
    file_handle.write('{:.6f}\t'.format(ave_max_distance))
    file_handle.write('{:.6f}\t'.format(ave_final_distance))
    file_handle.write('{:.6f}\t'.format(min_final_distance))
    file_handle.write('{:.6f}\t'.format(max_final_distance))
    file_handle.write('{:.4f}\t\t'.format(ave_num_at_home))
    for ep in range(episodes):
        for idx in range(nr_agent):
            file_handle.write('{:.0f}\t'.format(step_arr[ep, idx]))
            file_handle.write('{:.4f}\t'.format(dis_arr[ep, idx]**0.5))
    file_handle.write('\n')
    file_handle.close()
