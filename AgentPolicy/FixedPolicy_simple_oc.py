
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

