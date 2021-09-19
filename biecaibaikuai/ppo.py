from elegantrl.run import *
from elegantrl.agent import AgentPPO
from elegantrl.env import PreprocessEnv
from elegantrl.envs.FinRL.StockTrading import StockTradingEnv, check_stock_trading_env
import gym
# from myenv import MyEnv
from device import getPic
from time import sleep
import torch
class MyEnv:  # environment wrapper
    def __init__(self):
        self.env_name = '我的自定义环境'
        self.state_dim = 2
        self.action_dim = 2 #action范围 (-1, 1)
        self.action_max = 1 #连续空间 action * action_max
        self.max_step = 10 #「最大训练步数」若每轮训练步数超过最大值，则强行终止，并判定为任务失败
        self.if_discrete = False # action_dim=1 if env.if_discrete else env.action_dim
        self.target_return = 2 ** 16 #目标奖励

    def reset(self):
        state = [1.0, 0.1]
        return state
    def step(self, action):
        print('action', action)
        #动作转坐标 标准差 （x,y） x=[a,a+375] y=[b,b+667]

        #点击
        sleep(0.5)
        #取图
        #网络输出state
        done = False # done会调用env.reset()
        state = [1.0, 0.1]
        if (action[0] > 0) :
            done = True
        return state, 1.0, done, dict()
        # return state, reward, done, dict()
if __name__ == '__main__':
    src_image = getPic()
    # gym.logger.set_level(40) # Block warning
    print('ss', src_image)
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()  # AgentSAC(), AgentTD3(), AgentDDPG()

    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
    # args.gamma = 0.95
    # args.rollout_num = 2 # the number of rollout workers (larger is not always faster)

    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.gamma = 0.98
    # args.if_per_or_gae = True

    args.agent.cri_target = True
    args.visible_gpu = '0'

    args.env = MyEnv()
    args.target_step = args.env.max_step * 4
    args.if_per_or_gae = True
    args.gamma = 0.98
    args.net_dim = 2 ** 7
    args.batch_size = 1#args.net_dim * 2
    train_and_evaluate(args)

    # train_and_evaluate_mp(args) # the training process will terminate once it reaches the target reward.