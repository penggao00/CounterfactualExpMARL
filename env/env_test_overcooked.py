from env.overcooked import *
from AgentPolicy.FixedPolicy_simple_oc import agent_fixedpolicy, agent1, agent2

if __name__ == '__main__':
    print("~~~~~~~~~~~~test env simple_overcooked~~~~~~~~~~~~~~~~~~~")
    env = SimpleOvercooked()
    env2 = SimpleOvercooked()
    p1=agent1(2,env)
    p2=agent2(opt=2,env=env2)
    env3 = SimpleOvercooked()
    env_cf=wrapped_env_cf(env3,[p1,p2],1)  # p2 -> 1(agent idx)
    print(env_cf.observe())

    obs={'agent_1': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'cord': [2, 0, 0], 'action_mask': [1., 1., 1., 1., 1.]}, 'agent_2': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'cord': [2, 3, 0], 'action_mask': [1., 1., 1., 1., 1.]}}

    print(env_cf.step(4))

    env_cf.set_state(obs)
    print(env_cf.observe())
    print(env_cf.get_actions(obs))
    # obs={'agent_1': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'cord': [2, 0, 0], 'action_mask': [1., 1., 1., 1., 1.]}, 'agent_2': {'observation': [[1, 7, 0, 0, -2], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [0, 9, 0, 0, 0], [2, 7, 0, 0, -1]], 'cord': [4, 4, 1], 'action_mask': [1., 1., 1., 1., 1.]}}
    print(env_cf.check_done(obs))







    # policy=agent_fixedpolicy(2,env)
    # obs_env=env.reset()
    # env.render()
    # print(obs_env)
    # while True:
    #     actions=[]
    #     for agent in env.agents:
    #         obs_a=obs_env[agent]
    #         # print(obs_a)
    #         mp=obs_a.get("observation")
    #         agent_info=obs_a.get("cord")
    #         mask=obs_a.get("action_mask")
    #         # print(mp,agent)
    #
    #         y=agent_info[0]
    #         x=agent_info[1]
    #         item=agent_info[0]
    #         if agent=="agent_1":
    #             act=policy.agent1(mp,agent_info)
    #         else:
    #             act=policy.agent2(mp,agent_info)
    #         actions.append(act)
    #     print(actions)
    #     obs_env,reward,done,info=env.step(actions)
    #     env.render()
    #     if done:
    #         print(obs_env,reward,done,info)
    #         break




