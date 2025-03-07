import pettingzoo
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv
import copy

from sympy import false
from wandb.cli.cli import agent

# 1 and 2 means corresponding materials; food 1 should deliver to where -1 is; food 2 should deliver to where -2 is.
# 9 is wall; 7 is where to transfer material; to simplify it, when agent 1 reach 1 or 2, immediately, change the neiborgh grid which is 7.
# then agent 1 cannot move; agent 2 needs to move towards the transfer grid once to attain the food.
original_map=[
[1,7,0,0,-2],
[0,9,0,0,0],
[0,9,0,0,0],
[0,9,0,0,0],
[2,7,0,0,-1],
]
simple_cook1={"map":original_map,"agent_1":[2,0,0],"agent_2":[2,3,0]}
# map and agent[cord_y, cord_x, item]

action_mp=[[0,0],[0,1],[1,0],[0,-1],[-1,0]]

class SimpleOvercooked(ParallelEnv):
    def __init__(self):
        super().__init__()
        self.agents = ["agent_1", "agent_2"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}  # 5 actions


        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0, high=255, shape=(5,5), dtype=np.uint8),  #map
                "agent":spaces.Box(low=0,high=5,shape=(1,1,1),dtype=np.uint8), #cord_y,cord_x,item
                "action_mask": spaces.MultiBinary(5),  # Binary mask for 5 actions
            }) for agent in self.agents
        }


    def get_action_mask(self, agent):
        """Define which actions are valid (1) or invalid (0)."""
        valid_actions = np.ones(5)  # Example mask (0 means invalid)
        if self.obs["agent_2"][2]!=0 or (self.obs.get("map")[0][0]!=1 or self.obs.get("map")[4][0]!=2):
            # item 1 or 2 no longer stay in the innitial state or in agent2's pocket: agent1 move ,agent 2 stay
            flag=True
        else:
            #otherwise: agent1 stay
            flag=False
        if agent=="agent_1":
            if flag:
                valid_actions=np.zeros(5)
                # after attain item, the agent 1 could only move towards left to put thing on transport
                valid_actions[1]=1
                valid_actions[0] = 1
        elif agent=="agent_2":
            valid_actions=valid_actions # agent2 always move
        return valid_actions


    def reset(self, seed=None, options=None):
        self.size_y = 5
        self.size_x = 5
        self.obs=copy.deepcopy(simple_cook1)
        self.done=False
        self.agents = self.possible_agents[:]
        ret={}
        for agent in self.agents:
            ret[agent]={"observation": self.obs.get("map"),
                        "cord":self.obs.get(agent),
                        "action_mask": self.get_action_mask(agent)}
        return ret
    def observe(self):
        ret={}
        for agent in self.agents:
            ret[agent]={"observation": self.obs.get("map"),
                        "cord":self.obs.get(agent),
                        "action_mask": self.get_action_mask(agent)}
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
            if self.get_action_mask(agent)[act]==0:
                continue
            print(action_mp)

            tmp_y = y+ action_mp[act][0]
            tmp_x = x + action_mp[act][1]
            print(y,x,tmp_y,tmp_x,action_mp[actions[i]][0],action_mp[actions[i]][1])
            if tmp_y<0 or tmp_y>=self.size_y or tmp_x<0 or tmp_x>=self.size_x:
                # come across boundry
                print(tmp_x, tmp_y)
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

env = SimpleOvercooked()
