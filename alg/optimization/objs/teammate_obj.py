from alg.optimization.objs.abs_obj import AbstractObj


class TeammateObj(AbstractObj):
#manipunate your teammate action; modified based on fid_obj
    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        max_actions = params['max_actions']
        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']
        others, other_idx=env.extract_others()
        for idx in other_idx:
            self.other=others[idx]
        super(TeammateObj, self).__init__(env, bb_model, params, max_actions)
        self.lmbdas = {'cost': -1,
               'reachability': -1,
               'stochastic_validity': -1,
               'fidelity': -1,   # past -1
               'validity': 0,
               'realistic': -0,
               'actionable': -0}


    def get_ind_unweighted_rews(self, fact, cf, target_action, actions, cummulative_reward):
        objectives = self.get_objectives(fact, cf, actions, target_action)

        rewards = {}
        total_rew = 0.0

        for o_name, o_formula in objectives.items():
            # print(o_name,o_formula)
            rewards[o_name] = -o_formula
            total_rew += -rewards[o_name]

        return rewards, total_rew
    def get_ind_weighted_rews(self, fact, cf, target_action, actions, cummulative_reward):
        objectives = self.get_objectives(fact, actions, cummulative_reward, target_action)

        rewards = {}
        total_rew = 0.0

        for o_name, o_formula in objectives.items():
            rewards[o_name] = self.lmbdas[o_name] * o_formula(cf)
            total_rew += rewards[o_name]

        return rewards, total_rew

    def get_reward(self, fact, cf, target_action, actions, cummulative_rew):
        objectives = self.get_objectives(fact, cf, actions, target_action)

        final_rew = 0.0
        # print(self.get_objectives(fact, cf, actions, target_action))
        # print(objectives,self.lmbdas)

        for o_name, o_formula in objectives.items():
            final_rew += self.lmbdas[o_name] * o_formula

        return final_rew

    # def get_objectives(self, fact, cf, actions, target_action):
    #     stochasticity, validity, cost, fidelity = self.calculate_stochastic_rewards(fact, cf, actions, target_action)
    #
    #     reachability = self.reachability(actions)
    #
    #     return {
    #         'cost': cost,
    #         'reachability': reachability,
    #         'stochasticity': stochasticity,
    #         'validity': validity
    #     }
    def get_objectives(self, fact, cf, actions, target_action):
        stochasticity, validity, cost, fidelity = self.calculate_stochastic_rewards(fact, actions, target_action, self.bb_model)

        reachability = self.reachability(actions)
        # print(reachability)
        ret = {
            'fidelity': fidelity,
            'reachability': reachability,
            'stochastic_validity': stochasticity
        }
        # print(ret)

        return ret

    def update_weights(self, w):
        self.lmbdas.update(w)

    def reachability(self, actions):
        if len(actions) == 0:
            return 1
        # print("reach",actions, self.max_actions)

        return len(actions) * 1.0 / self.max_actions

    def calculate_stochastic_rewards(self, fact, actions, target_action, bb_model):
        # run simulations from fact with actions
        n_sim = self.n_sim

        diff_outcomes = {}
        target_outcome = 0.0
        total_cost = 0.0

        fidelities = []

        for s in range(n_sim):
            self.env.reset()
            self.env.set_state(fact)

            obs = fact

            fid = 0.0

            if len(actions) == 0:
                return 1, 1, 1, 1

            done = False
            early_break = False
            available_actions = self.env.get_actions(fact)
            ep_rew = 0.0
            for a in actions:
                if done or (a not in available_actions) or (len(available_actions) == 0):
                    early_break = True
                    break

                prob = bb_model.get_action_prob(obs, a)
                # teammate_view = self.env.opponent_view(obs)
                # prob=self.other.get_action_prob(teammate_view,a)
                fid += prob

                obs, rew, done, _ = self.env.step(a)
                ep_rew += rew

                available_actions = self.env.get_actions(obs)

            if not early_break:
                # count how many different actions are chosen after the path

                # outcome = self.bb_model.predict(obs)
                teammate_view = self.env.opponent_view(obs)
                # print("cal_stoc_reward")
                outcome=self.other.predict(teammate_view)

                if outcome in list(diff_outcomes.keys()):
                    diff_outcomes[outcome] += 1
                else:
                    diff_outcomes[outcome] = 1

                target_outcome += (outcome == target_action)

                fidelities.append(1 - fid/len(actions))

            total_cost += ep_rew

        # print("diff_out",len(diff_outcomes))

        if len(diff_outcomes):
            most_freq_outcome = max(list(diff_outcomes.values()))
            stochasticity = 1 - (most_freq_outcome * 1.0 / n_sim)
            # print(stochasticity)
        else:
            stochasticity = 1

        validity = 1 - target_outcome / n_sim
        cost = (total_cost / n_sim) / (self.max_actions * self.env.max_penalty)
        if len(fidelities):
            fidelity = sum(fidelities) / (len(fidelities) * 1.0)
        else:
            fidelity = 1

        return stochasticity, validity, cost, fidelity