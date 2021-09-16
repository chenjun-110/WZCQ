from elegantrl.run import *
from elegantrl.agent import AgentPPO
from elegantrl.env import PreprocessEnv
import gym
if __name__ == '__main__':
    # gym.logger.set_level(40) # Block warning

    # args = Arguments(if_on_policy=True)
    # args.agent = AgentPPO()  # AgentSAC(), AgentTD3(), AgentDDPG()
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
    # args.gamma = 0.95
    # args.rollout_num = 2 # the number of rollout workers (larger is not always faster)

    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = '0'

    args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    args.target_step = args.env.max_step * 4
    args.if_per_or_gae = True
    args.gamma = 0.98

    train_and_evaluate(args)

    # train_and_evaluate_mp(args) # the training process will terminate once it reaches the target reward.