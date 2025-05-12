from alg.baselines.abs_baseline import AbstractBaseline
from alg.models.counterfactual import CF
from alg.optimization.objs.teammate_obj import TeammateObj
from alg.optimization.search.heuristic_tree_search_teammate_seqact import HeuristicTreeSearch

import numpy as np

class TMoRACCER(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = TeammateObj(env, bb_model, params)
        self.optim = HeuristicTreeSearch(env, bb_model, self.obj, params)
        self.env=env

        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']

        super(TMoRACCER, self).__init__()

    def generate_counterfactuals(self, fact, target):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target):
        ''' Returns all cfs found in the tree '''
        cfs = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        if len(cfs):
            # print("cf example",cfs[0])
            cfs = [CF(cf[0], cf[1], cf[2], cf[3]) for cf in cfs]
            cf_values = [cf.value for cf in cfs]

            best_value = max(cf_values)
            best_cfs = [cf for cf in cfs if cf.value == best_value]

            best_cf = self.choose_closest(best_cfs, fact)
            return best_cf
        else:
            return None

    def choose_closest(self, cfs, fact):
        diffs = []
        # print(len(cfs),cfs)
        for c in cfs:

            cf=self.env.flatten_state_noMask(c.cf)
            fact_flatten=self.env.flatten_state_noMask(fact)
            d=np.sum(cf!=fact_flatten)
            diffs.append(d)
            # print("!!",c.cf, d)

        min_diff_index = np.argmax(diffs)
        # min_diff_index = np.argmin(diffs)
        return cfs[min_diff_index]