from time import sleep
import torch
class MyEnv:  # environment wrapper
    def __init__(self):
        self.env_name = '我的自定义环境'
        self.state_dim = 1
        self.action_dim = 2 #action范围 (-1, 1)
        self.action_max = 0 #连续空间 action * action_max
        self.max_step = 10 #「最大训练步数」若每轮训练步数超过最大值，则强行终止，并判定为任务失败
        self.if_discrete = False # action_dim=1 if env.if_discrete else env.action_dim
        self.target_return = 2 ** 16 #目标奖励

    def reset(self):
        state = torch.tensor([1.0])
        return state
    def step(self, action):
        #动作转坐标 标准差 （x,y） x=[a,a+375] y=[b,b+667]

        #点击
        sleep(0.5)
        #取图
        #网络输出state
        done = False
        return torch.tensor([1.0]), 1.0, done, dict()
        # return state, reward, done, dict()