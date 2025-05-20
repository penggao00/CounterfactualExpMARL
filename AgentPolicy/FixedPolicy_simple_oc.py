from nbformat.v2 import downgrade

action_mp=[[0,0],[0,1],[1,0],[0,-1],[-1.0]]

class agent_fixedpolicy():
    def __init__(self,opt,env):
        # opt should be 4 or 2
        self.env=env
        self.opt=opt
    def agent1(self,mp,agent1):
        if agent1[0]==0 or agent1[0]==self.env.size_y-1:
            if agent1[2]==0:
                return 0
            else:
                return 1 # right when at collecting point
        else:
            return self.opt

    def agent2(self,mp,agent2):
        if agent2[2]==0:
            if mp[0][1]!=7:
                if agent2[0]!=0:
                    return 4 # up
                else:
                    return 3 #left
            elif mp[self.env.size_y-1][1]!=7:
                if agent2[0]!=self.env.size_y-1:
                    return 2 # down
                else:
                    return 3 #left
        else:
            if agent2[2]==1:
                if agent2[0]<self.env.size_y-1:
                    return 2 # move down
                else:
                    return 1 # move right when at bottom
            elif agent2[2]==2:
                if agent2[1]>0:
                    return 4 # move up
                else:
                    return 1 # move right when at bottom
        return 0

class agent1:
    def __init__(self,opt,env):
        self.model=agent_fixedpolicy(opt,env)
    def predict(self, x):
        x = x["agent_1"]
        action = self.model.agent1(x.get("observation"),x.get("cord"))
        return action

    def get_action_prob(self,obs, a):
        if self.predict(obs)==a:
            return 1
        else:
            return 0

class agent2:
    def __init__(self, opt, env):
        self.model = agent_fixedpolicy(opt, env)

    def predict(self, x):
        x=x["agent_2"]
        action = self.model.agent2(x.get("observation"),x.get("cord"))
        return action
    def get_action_prob(self,obs, a):
        if self.predict(obs)==a:
            return 1
        else:
            return 0

class agent1_fix:
    def __init__(self,opt=4):
        self.opt=opt
    def extract(self,x):
        self.agent_name=x["agent_name"]
        if self.agent_name!="agent_1":
            assert "wrong agentss"
        for agent in x["others"]:
            if agent!=self.agent_name:
                self.teammate=x["others"][agent]
        self.info=x["agent"]
        self.map=x["observation"]

    def predict(self, x):
        self.extract(x)
        agent1 = self.info
        if self.info[0]==0 or self.info[0]==len(self.map)-1:
            if agent1[2]==0:
                return 0
            else:
                return 1 # right when at collecting point
        else:
            return self.opt

    def get_action_prob(self,obs, a):
        if self.predict(obs)==a:
            return 1
        else:
            return 0

class agent2_fix:
    def __init__(self,opt=4):
        self.opt=opt
    def extract(self,x):
        self.agent_name=x["agent_name"]
        # self.agent_name=x["agent_name"]
        if self.agent_name!="agent_2":
            assert "wrong agentss"
        for agent in x["others"]:
            if agent!=self.agent_name:
                self.teammate=x["others"]["agent_1"]
        self.info=x["others"]["agent_2"]
        self.map=x["observation"]

        # print(self.agent_name,self.teammate,self.info,x)

    def predict(self, x):  #action_mp=[[0,0],[0,1],[1,0],[0,-1],[-1,0]] stay, right, down, left, up
        self.extract(x)
        agent2=self.info
        l=len(self.map)
        if agent2[2]==0:
            if self.map[0][1]==7 and self.map[l-1][1]==7:
                # no item on the port, follow the other agent
                if self.teammate[0]==l-self.teammate[0]-1:
                    return 0

                if self.teammate[0]<l-self.teammate[0]-1:
                    return 4 #up
                else:
                    return 2 #down
            elif self.map[0][1]!=7 and self.map[l-1][1]==7:
                #if [0,1] have item, go up then left
                if agent2[0]!=0:
                    return 4 # up
                else:
                    return 3 #left
            elif self.map[0][1]==7 and self.map[l-1][1]!=7:
                # [l-1,1] have item, go down then left
                if agent2[0] != l - 1:
                    return 2  # down
                else:
                    return 3  # left
            elif self.map[0][1]!=7 and self.map[l-1][1]!=7:
                if agent2[0]<l-agent2[0]-1:
                    # up is closer, then go up
                    ret=4
                else:
                    # down is closer, then go down
                    ret=2
                if agent2[0]==0 or agent2[0]==l-1:
                    # if agent is next to the port, go left to collect item.
                    ret=3
                return ret


            # if self.map[0][1]!=7 or self.teammate[0]<l-self.teammate[0]-1:
            #     if agent2[0]!=0:
            #         return 4 # up
            #     else:
            #         return 3 #left
            # elif self.map[l-1][1]!=7 or self.teammate[0]>l-self.teammate[0]-1:
            #     if agent2[0]!=l - 1:
            #         return 2 # down
            #     else:
            #         return 3 #left
        else:
            if agent2[2]==1:
                # if agent2[0]>0:
                if agent2[0]<l-1:
                    return 2 # move down
                else:
                    return 1 # move right when at bottom
            elif agent2[2]==2:
                # if agent2[0]<len(self.map)-1:
                if agent2[1]<l-1:
                    return 1 # move right when at bottom
                else:
                    return 4 # move up


        return 0
    def get_action_prob(self,obs, a):
        if self.predict(obs)==a:
            return 1
        else:
            return 0
