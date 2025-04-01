
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
                if agent2[0]>0:
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
        for agent in x["others"]:
            if agent!=self.agent_name:
                self.teammate=x["others"][agent]
        self.info=x["agent"]
        self.map=x["observation"]

    def predict(self, x):
        self.extract(x)
        agent1 = self.info
        if self.info[0]==0 or self.info[0]==len(self.map):
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
        for agent in x["others"]:
            if agent!=self.agent_name:
                self.teammate=x["others"][agent]
        self.info=x["agent"]
        self.map=x["observation"]

    def predict(self, x):
        self.extract(x)
        agent2=self.info
        l=len(self.map)
        if agent2[2]==0:
            if self.map[0][1]!=7 or self.teammate[0]<l-self.teammate[0]:
                if agent2[0]!=0:
                    return 4 # up
                else:
                    return 3 #left
            elif self.map[l-1][1]!=7 or self.teammate[0]>l-self.teammate[0]:
                if agent2[0]!=l - 1:
                    return 2 # down
                else:
                    return 3 #left
        else:
            if agent2[2]==1:
                if agent2[0]<len(self.map)-1:
                    return 2 # move down
                else:
                    return 1 # move right when at bottom
            elif agent2[2]==2:
                if agent2[0]>0:
                    return 4 # move up
                else:
                    return 1 # move right when at bottom
        return 0
    def get_action_prob(self,obs, a):
        if self.predict(obs)==a:
            return 1
        else:
            return 0
