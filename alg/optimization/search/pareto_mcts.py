from alg.optimization.algs.heuristic_ts_alg import HeuristicTSAlgorithm
from alg.optimization.algs.pareto_mcts import ParetoMCTS
from alg.optimization.search.tree_search import TreeSearch


class ParetoMCTSSearch(TreeSearch):

    def __init__(self, env, bb_model, obj, params):
        alg = ParetoMCTS(env, bb_model, obj, params)
        super(ParetoMCTSSearch, self).__init__(env, bb_model, obj, params, alg)


