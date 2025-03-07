from env.overcooked import *
from AgentPolicy.FixedPolicy_simple_oc import agent_fixedpolicy

if __name__ == '__main__':
    print("~~~~~~~~~~~~test env simple_overcooked~~~~~~~~~~~~~~~~~~~")
    env = SimpleOvercooked()
    policy=agent_fixedpolicy(2,env)
    obs_env=env.reset()
    env.render()
    print(obs_env)
    while True:
        actions=[]
        for agent in env.agents:
            obs_a=obs_env[agent]
            # print(obs_a)
            mp=obs_a.get("observation")
            agent_info=obs_a.get("cord")
            mask=obs_a.get("action_mask")
            # print(mp,agent)

            y=agent_info[0]
            x=agent_info[1]
            item=agent_info[0]
            if agent=="agent_1":
                act=policy.agent1(mp,agent_info)
            else:
                act=policy.agent2(mp,agent_info)
            actions.append(act)
        print(actions)
        obs_env,reward,done,info=env.step(actions)
        env.render()
        if done:
            print(obs_env,reward,done,info)
            break


