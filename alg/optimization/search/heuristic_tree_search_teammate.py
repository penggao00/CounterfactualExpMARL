from alg.optimization.algs.heuristic_ts_alg_teammate import HeuristicTSAlgorithmTM
from alg.optimization.search.tree_search import TreeSearch


class HeuristicTreeSearch(TreeSearch):

    def __init__(self, env, bb_model, obj, params):
        alg = HeuristicTSAlgorithmTM(env, bb_model, obj, params)
        super(HeuristicTreeSearch, self).__init__(env, bb_model, obj, params, alg)


