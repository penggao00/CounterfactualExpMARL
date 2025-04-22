import json

from sympy.physics.vector.printing import params

# from alg.baselines.fid_raccer import (FidRACCER)
# from alg.baselines.ganterfactual import GANterfactual
# from alg.baselines.mcts_raccer import MCTSRACCER
from alg.baselines.teammateObj_raccer import TMoRACCER
# from src.envs.farm0 import Farm0
# from src.envs.frozen_lake import FrozenLake
# from src.envs.gridworld import Gridworld
from alg.evaluation.eval import evaluate_objectives, get_realistic_df, split_df, print_summary_split, evaluate_all
# from alg.models.bb_model import BBModel
# from alg.optimization.objs.fid_obj import FidObj
# from alg.optimization.objs.game_obj import GameObj
# from alg.optimization.objs.sl_obj import SLObj
from alg.optimization.objs.teammate_obj import TeammateObj
# from src.tasks.task import Task
# from src.utils.utils import seed_everything, load_facts_from_summary, load_facts_from_csv, generate_summary_states
# from env.overcooked2 import *
from AgentPolicy.FixedPolicy_simple_oc import agent_fixedpolicy

from os.path import exists

import pandas as pd
from tqdm import tqdm

#
# class Task:
#
#     def __init__(self, task_name, env, bb_model, method, method_name, eval_path, eval_objs, params):
#         self.task_name = task_name
#         self.env = env
#         self.bb_model = bb_model
#         self.method = method
#         self.method_name = method_name
#         self.eval_path = eval_path
#         self.params = params
#
#         self.eval_obj = eval_objs
#
#     def run_experiment(self, facts, targets=None):
#         print('Running experiment for {} task with {}'.format(self.task_name, self.method_name))
#         print('Finding counterfactuals for {} facts'.format(len(facts)))
#
#         # get cfs for facts
#         cnt = 0
#
#         for i in tqdm(range(len(facts))):  # for each fact
#             f = facts[i]
#
#             if isinstance(f, dict):
#                 f = self.env.generate_state_from_json(f)
#
#             if targets is None:
#                 # if target is not given, go through all possible targets
#                 ts = self.get_targets(f, self.env, self.bb_model)
#             else:
#                 ts = targets[i]
#
#             for t in ts:
#                 cfs = self.method.generate_counterfactuals(f, t)
#
#                 if cfs is None:
#                     found = False
#                     self.evaluate_cf(f, t, cfs, found, cnt)
#                     continue
#                 else:
#                     if not isinstance(cfs, list):
#                         cfs = [cfs]
#
#                     for cf in cfs:
#                         found = True
#                         self.evaluate_cf(f, t, cf, found, cnt)
#
#             cnt += 1
#
#     def get_targets(self, f, env, bb_model):
#         pred = bb_model.predict(f)
#         available_actions = env.get_actions(f)
#         targets = [a for a in available_actions if a != pred]
#
#         return targets
#
#     def evaluate_cf(self, f, t, cf, found, fact_id):
#         eval_obj_names = []
#
#         for eo in self.eval_obj:
#             eval_obj_names += eo.objectives
#
#         if not found:
#             df_values = {e: -1 for e in eval_obj_names}
#
#             df_values['recourse'] = 0
#             df_values['cf_readable'] = 0
#             df_values['cf'] = 0
#         else:
#             df_values = cf.reward_dict
#
#             recourse = str(cf.recourse).split('[')[1].split(']')[0]
#             if recourse == '':
#                 recourse = self.generate_recourse(f, cf.cf, self.env, self.params['max_actions'])
#                 if recourse != 0:
#                     cf.recourse = recourse
#                 recourse = str(recourse)
#
#             for eo in self.eval_obj:
#                 obj_vals = eo.get_objectives(f, cf.cf, cf.recourse, t)
#                 for o_name, o_val in obj_vals.items():
#                     if o_name not in list(df_values.keys()):  # if not already evaluated
#                         df_values[o_name] = o_val
#
#             df_values['recourse'] = recourse
#             df_values['cf_readable'] = self.env.writable_state(cf.cf)
#             df_values['cf'] = [list(cf.cf)]
#
#         df_values['fact'] = [list(f)]
#         df_values['fact_readable'] = self.env.writable_state(f)
#         df_values['target'] = t
#         df_values['fact_id'] = fact_id
#
#         header = not exists(self.eval_path)
#         if not header:
#             old_df = pd.read_csv(self.eval_path, header=0)
#             df = pd.concat([old_df, pd.DataFrame(df_values,index=[0])], ignore_index=True)
#             df.to_csv(self.eval_path, mode='w', index=None)
#         else:
#             df = pd.DataFrame(df_values, index=[0])
#             df.to_csv(self.eval_path, mode='a', header=header, index=None)
#
#     def generate_recourse(self, fact, cf, env, max_actions):
#         states = []
#
#         states.append((fact, []))
#         level = 0
#         expand = 10
#         done = False
#
#         while level <= max_actions and len(states) > 0 and not done:
#             curr_state, curr_actions = states[-1]
#             states = states[:-1]
#
#             env.reset()
#             env.set_state(curr_state)
#             available_actions = env.get_actions(curr_state)
#
#             for a in available_actions:
#                 for e in range(expand):
#                     env.set_state(curr_state)
#                     new_state, rew, done, _ = env.step(a)
#
#                     states.append((new_state, curr_actions + [a]))
#
#                     if env.equal_states(new_state, cf):
#                         return curr_actions + [a]
#
#             level += 1
#
#         return 0
#

#
#
#
# def main(task_name, agent_type):
#     print('TASK = {} AGENT_TYPE = {}'.format(task_name, agent_type))
#
#     training_timesteps = 300000
#     if agent_type == 'suboptim':
#         training_timesteps = training_timesteps / 10
#
#     env = SimpleOvercooked()
#     gym_env=env
#     bb_model=agent_fixedpolicy
#     # bb_model = BBModel(gym_env, model_path, training_timesteps)
#
#     # load parameters
#
#     params={
#     "layer_shapes": [[32, 512], [512, 512], [512, 26]],
#     "ts_n_iter": 300,
#     "ts_n_expand": 20,
#     "n_sim": 10,
#     "max_actions": 20,
#     "max_level": 20,
#     "c": 0.7
#     }
#
#
#
#
#     # get facts
#     # facts, targets = load_facts_from_csv(fact_csv_dataset_path, env, bb_model)
#
#     # facts, targets = load_facts_from_json(fact_json_path)
#     # facts, targets = load_facts_from_summary(env, bb_model)
#     # one fact lookesl ike{"agent": 1,
#     #  "monster": 13,
#     #  "trees": [{"2": 5}, {"7": 3}, {"12": 3}, {"17": 3}, {"22": 5}],
#     #  "target": 5
#     #  }
#     # define methods
#     fid_raccer = FidRACCER(env, bb_model, params)
#     # mcts_raccer = MCTSRACCER(env, bb_model, params)
#     # ganterfactual = GANterfactual(env, bb_model, params, generator_path)
#
#     methods = [fid_raccer]
#     method_names = ['FidRACCER']
#
#     # define objectives
#     sl_obj = SLObj(env, bb_model, params)
#     game_obj = GameObj(env, bb_model, params)
#     fid_obj = FidObj(env, bb_model, params)
#
#     # define eval objectives
#     eval_objs = [[sl_obj, game_obj, fid_obj], [sl_obj, game_obj, fid_obj]]
#
#     # for i, m in enumerate(methods):
#     #     print('\n------------------------ {} ---------------------------------------\n'.format(method_names[i]))
#     #     eval_path = 'eval/{}/{}/{}'.format(task_name, method_names[i], agent_type)
#     #     task = Task(task_name, env, bb_model, m, method_names[i], eval_path, eval_objs[i], params)
#     #
#     #     task.run_experiment(facts[1:],  targets)
#
#
#     pred = bb_model.predict(f)
#     available_actions = env.get_actions(f)
#     t = [a for a in available_actions if a != pred]
#
#     cfs = fid_raccer.generate_counterfactuals(f, t)
#
#     # if cfs is None:
#     #     found = False
#     #     self.evaluate_cf(f, t, cfs, found, cnt)
#     #     continue
#     # else:
#     #     if not isinstance(cfs, list):
#     #         cfs = [cfs]
#     #
#     #     for cf in cfs:
#     #         found = True
#     #         self.evaluate_cf(f, t, cf, found, cnt)
#
#     evaluate_all(tasks=['frozen_lake', 'gridworld', 'farm0'],
#                  agent_types=['optim', 'suboptim', 'non_optim'],
#                  method_names=method_names,
#                  eval_objs=[sl_obj, game_obj, fid_obj])
#
#     # eval_path_template = 'eval/{}/{}/{}'
#     # eval_paths = [eval_path_template.format(task_name, method_name, agent_type) for method_name in method_names]
#     # realistic_df = get_realistic_df(eval_paths, targets=[4, 5])
#     # summary = generate_summary_states(env, bb_model, realistic_df)
#     # indices = split_df(summary)
#     # #
#     # print_summary_split(summary, *indices, eval_paths, targets=[4, 5])


if __name__ == '__main__':
    # main('farm0', 'optim')
    # main('farm0', 'suboptim')

    # main('gridworld', 'optim')
    # main('gridworld', 'non_optim')
    # main('gridworld', 'suboptim')
    #
    # main('frozen_lake', 'optim')
    # main('frozen_lake', 'suboptim')
    from env.overcooked2 import *
    from AgentPolicy.FixedPolicy_simple_oc import agent1_fix,agent2_fix

    p1=agent1_fix(2)
    p2=agent2_fix(opt=2)
    env3 = SimpleOvercooked()
    env_cf=wrapped_env_cf(env3,[p1,p2],0)  # p2 -> 1(agent idx)
    print(env_cf.extract_others())

    # f={'agent_1': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'agent': [2, 0, 0], 'action_mask': [1., 1., 1., 1., 1.]}, 'agent_2': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'cord': [2, 3, 0], 'action_mask': [1., 1., 1., 1., 1.]}}
    f=env_cf.reset()
    print("2323",f)
    print("opponent",env_cf.opponent_view())
    # draw(f)
    t=1

    params={
    # "layer_shapes": [[32, 512], [512, 512], [512, 26]],
    "ts_n_iter": 5000,
    "ts_n_expand":5,
    "n_sim": 10,
    "max_actions": 20,
    "max_level": 30,
    "c": 0.7
    }
    tmo_raccer = TMoRACCER(env_cf, p2, params)
    # cfs=tmo_raccer.optim.alg.search(init_state=f, fact=f, target_action=t)
    cfs = tmo_raccer.generate_counterfactuals(f, t)
    # for cf in cfs:
    #     print(cf)

    print("target", t, cfs.cf)
    print([[0,0],[0,1],[1,0],[0,-1],[-1,0]])

    draw(cfs.cf)


    # draw(cfs.cf)
    # for cf in cfs:
    #     found = True
    #     evaluate_cf(f, t, cf, found, cnt)




