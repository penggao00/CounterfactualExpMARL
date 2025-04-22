from typing import Tuple

import gym
import pettingzoo
from gym.core import ObsType
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv
import copy

from sympy import false
from wandb.cli.cli import agent

# 1 and 2 means corresponding materials; food 1 should deliver to where -1 is; food 2 should deliver to where -2 is.
# 9 is wall; 7 is where to transfer material; to simplify it, when agent 1 reach 1 or 2, immediately, change the neiborgh grid which is 7.
# then agent 1 cannot move; agent 2 needs to move towards the transfer grid once to attain the food.
original_map=np.array([
[1,7,0,0,-2],
[0,9,0,0,0],
[0,9,0,0,0],
[0,9,0,0,0],
[2,7,0,0,-1],
])
simple_cook1={"map":original_map,"agent_1":np.array([2,0,0]),"agent_2":np.array([2,2,0])}
# map and agent[cord_y, cord_x, item]

action_mp=np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]])
class SimpleOvercooked(ParallelEnv):
    def __init__(self):
        super().__init__()
        self.agents = ["agent_1", "agent_2"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 5 actions

        self.max_penalty=-1   # a parameter needed in original code, it works only in FrozenLake?


        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=255, shape=(5,5), dtype=np.uint8),  #map
                "agent":spaces.Box(low=0,high=5,shape=(1,1,1),dtype=np.uint8), #cord_y,cord_x,item
                "action_mask": spaces.MultiBinary(5),  # Binary mask for 5 actions
            }) for agent in self.agents
        }
        self.reset()


    def get_action_mask(self, agent):
        """Define which actions are valid (1) or invalid (0)."""
        valid_actions = np.ones(5)  # Example mask (0 means invalid)
        # if self.obs["agent_2"][2]!=0 or (self.obs.get("map")[0][0]!=1 or self.obs.get("map")[4][0]!=2):
        #     # item 1 or 2 no longer stay in the innitial state or in agent2's pocket: agent1 move ,agent 2 stay
        #     flag=True
        # else:
        #     #otherwise: agent1 stay
        #     flag=False
        # if agent=="agent_1":
        #     if flag:
        #         valid_actions=np.zeros(5)
        #         # after attain item, the agent 1 could only move towards left to put thing on transport
        #         valid_actions[1]=1
        #         valid_actions[0] = 1
        # elif agent=="agent_2":
        #     valid_actions=valid_actions # agent2 always move
        return valid_actions


    def reset(self, seed=None, options=None):
        self.size_y = 5
        self.size_x = 5
        self.obs=copy.deepcopy(simple_cook1)
        self.done=False
        self.agents = self.possible_agents[:]

        return self.observe()
    def observe(self):
        others={}
        for agent in self.agents:
            others[agent]=self.obs.get(agent)
        ret={}
        for agent in self.agents:
            ret[agent]={"observation": self.obs.get("map"),
                        "agent":self.obs.get(agent),
                        "action_mask": self.get_action_mask(agent),
                        "agent_name":agent,
                        "others":others}
        return ret

    def step(self, actions):
        reward=0
        if self.done==True:
            return self.observe(), 0, self.done, {}  # obs, reward, terminal, info

        if len(self.agents)!=len(actions):
            assert "input actions should have the same length as agent_list"
        for i in range(0,len(actions)):
            agent=self.agents[i]
            y = self.obs.get(agent)[0]
            x = self.obs.get(agent)[1]
            item = self.obs.get(agent)[2]
            act=actions[i]
            # print(self.get_action_mask(agent),act)
            if self.get_action_mask(agent)[act]==0:
                continue
            # print(action_mp)

            tmp_y = y+ action_mp[act][0]
            tmp_x = x + action_mp[act][1]
            # print(y,x,tmp_y,tmp_x,action_mp[actions[i]][0],action_mp[actions[i]][1])
            if tmp_y<0 or tmp_y>=self.size_y or tmp_x<0 or tmp_x>=self.size_x:
                # come across boundry
                # print(tmp_x, tmp_y)
                continue
            # print(tmp_x,tmp_y)
            if self.obs.get('map')[tmp_y][tmp_x]==9:
                # wall
                continue
            elif (tmp_y==0 and tmp_x==1) or (tmp_y==self.size_y-1 and tmp_x==1):
                # two transport grid as 7
                if self.obs.get('map')[tmp_y][tmp_x]!=7:
                    # there exist item
                    if item==0:
                        # empty pocket, attain it.
                        self.obs.get(agent)[2] = self.obs.get('map')[tmp_y][tmp_x]
                        self.obs.get('map')[tmp_y][tmp_x] = 7
                        continue
                    else:
                        # pockek is full
                        continue

                else:
                    # empty transport ==7
                    if item!=0:
                        # put item on transport
                        self.obs.get('map')[tmp_y][tmp_x]=self.obs.get(agent)[2]
                        self.obs.get(agent)[2]=0
                        continue
                    else:
                        continue
            elif self.obs.get('map')[tmp_y][tmp_x]==1 or self.obs.get('map')[tmp_y][tmp_x]==2:
                # item grid to collect
                if item==0:
                    self.obs.get(agent)[2] = self.obs.get('map')[tmp_y][tmp_x]
                    self.obs.get('map')[tmp_y][tmp_x]=0
            # not above scenario, then move

            self.obs.get(agent)[0]=tmp_y
            self.obs.get(agent)[1]=tmp_x
            # print("update cord",agent, tmp_y,tmp_x)



        for agent in self.agents:
            y = self.obs.get(agent)[0]
            x = self.obs.get(agent)[1]
            item = self.obs.get(agent)[2]
            if self.obs.get("map")[y][x]+item==0 and item!=0:
                reward=1
                self.done=True

        return self.observe(), reward, self.done, {}  #obs, reward, terminal, info

    def render(self):
        print(self.obs)

    # def set(self,state):
        # for agent in state:
        #     s=state.get(agent)
        #     self.obs["map"]=s["observation"]
        #     self.obs[agent]=s["agent"]
    def check_done(self,state):
        done=False
        map=self.obs.get("map")
        for agent in self.agents:
            obs=state["others"][agent]
            y = obs[0]
            x = obs[1]
            item = obs[2]
            if map[y][x]+item==0 and item!=0:
                done=True
        return done
class wrapped_env_cf(gym.Wrapper):
    def __init__(self, env,agent_list,agent):
        super().__init__(env)
        self.agent_policy=agent_list
        self.idx=agent

    def opponent_view(self,state=None):
        id=1-self.idx
        agent=self.env.agents[id]
        if state is None:
            return self.env.observe()[agent]
        else:
            # print("set",state,"obs",self.obs)
            self.set_state(state)
            # print("after set",self.env.observe(),agent)
            return self.env.observe()[agent]

    def reset(self, seed=None, options=None):
        self.env.size_y = 5
        self.env.size_x = 5
        self.env.obs=copy.deepcopy(simple_cook1)
        self.env.done=False
        self.env.agents = self.possible_agents[:]
        ag=self.agents[self.idx]
        return self.env.observe()[ag]

    def step(self, action):
        ret = self.env.observe()
        actions=[]
        for agent in self.agents:
            i=self.env.agents.index(agent)
            obs=ret[agent]

            a=self.agent_policy[i].predict(obs)
            actions.append(a)
        # print(actions)
        actions[self.idx]=action
        # print(actions)
        # return self.env.step(actions)
        observe, reward, done, _ = self.env.step(actions)
        ag=self.agents[self.idx]
        return observe[ag],reward,done,_

    def get_actions(self,state):
        x=self.env.get_action_mask(self.agents[self.idx])
        l=len(x)
        ret=[]
        for i in range(0,l):
            if x[i]!=0:
                ret.append(i)
        return ret

    def check_done(self, state):
        done=False
        map=state.get("observation")
        for agent in self.agents:
            obs=state["others"][agent]
            y = obs[0]
            x = obs[1]
            item = obs[2]
            if map[y][x]+item==0 and item!=0 and map[y][x]<0:
                done=True
        return done

    def set_state(self,state):
        # print("set",state,self.obs,simple_cook1)
        # print(state["observation"],state["others"])
        self.env.obs["map"] = copy.deepcopy(state["observation"])
        for agent in state["others"]:
            # print(agent,          state["others"][agent])
            self.env.obs[agent] = copy.deepcopy(state["others"][agent])
        self.env.done=self.check_done(state)
        # print("after set state function",self.env.obs,self.env.observe())

    def realistic(self, x):
        ''' Returns a boolean indicating if x is a valid state in the environment (e.g. chess state without kings is not valid)'''
        return True

    def actionable(self, x, fact):
        ''' Returns a boolean indicating if all immutable features remain unchanged between x and fact states'''
        return True

    def flatten_state_noMask(self,state):
        ret=[]
        mp = np.array([item for row in state["observation"] for item in row])
        ret=mp
        for agent in self.agents:
            a= np.array(state["others"][agent])
            ret = np.concatenate((ret,a))

        return ret

    def extract_others(self):
        l=[]
        for i in range(0,len(self.agent_policy)):
            if i!=self.idx:
                l.append(i) #index list except controled one

        return self.agent_policy,l


    def equal_states(self, obs , state):
        eq=False
        # print(obs,state)
        obs=self.flatten_state_noMask(obs)
        state=self.flatten_state_noMask(state)
        # print(obs,state)
        eq= (obs==state).all()
        # print(eq)
        return eq
    def render(self):
        draw(self.opponent_view())

def draw(state):
    import matplotlib.pyplot as plt
    # Your provided data
    observation = np.array(state["observation"])

    # Agents' positions and items (row, column, item)
    print(state)
    agents = state['others']
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the grid world
    # cmap = plt.cm.coolwarm
    # ax.imshow(observation, cmap=cmap, interpolation='none',origin='upper')
    #
    # # Add annotations (cell values)
    # for (j, i), val in np.ndenumerate(observation):
    #     ax.text(i, j, val, ha='center', va='center', fontsize=12, color='black')
    #
    # # Mark agent positions and items
    # for agent_name, (row, col, item) in agents.items():
    #     ax.scatter(col, row, marker='o', s=300, label=agent_name, edgecolors='black', linewidths=1.5)
    #     ax.text(col, row, agent_name, ha='center', va='center', color='white', fontsize=9)
    #     if item != 0:
    #         ax.text(col, row + 0.3, f'Item: {item}', ha='center', va='center', color='yellow', fontsize=7,
    #                 fontweight='bold')

    cmap = plt.cm.coolwarm
    ax.imshow(observation, cmap=cmap, interpolation='none', origin='upper')
    l=5-1


    # Add annotations (cell values)
    for (j, i), val in np.ndenumerate(observation):
        ax.text(i, l-j, val, ha='center', va='center', fontsize=12, color='black')

    # Mark agent positions and items
    for agent_name, (row, col, item) in agents.items():
        print(agent_name,row,col,item)
        ax.scatter(col, l-row, marker='o', s=300, label=agent_name, edgecolors='black', linewidths=1.5)
        ax.text(col, l-row, agent_name, ha='center', va='center', color='white', fontsize=9)
        if item != 0:
            ax.text(col, l-row + 0.3, f'Item: {item}', ha='center', va='center', color='yellow', fontsize=7,
                    fontweight='bold')

    #

    # Set up grid lines
    ax.set_xticks(np.arange(-0.5, observation.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, observation.shape[0], 1))
    ax.grid(color='black', linestyle='-', linewidth=1)

    # Remove axis labels for clarity
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title('Grid World Visualization with Agents and Items', fontsize=14)
    # plt.legend(loc='upper right')
    plt.gca().invert_yaxis()
    plt.show()
