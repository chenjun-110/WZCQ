# cartpole文件在   环境文件夹\Lib\site-packages\gym\envs\classic_control
# gym安装：pip install gym matplotlib -i  https://pypi.tuna.tsinghua.edu.cn/simple
# 如要迁移别的游戏，修改net状态数，动作数，记忆库存储列数，奖励算法
import random
import torch
import torch.nn as nn
import numpy as np
import gym

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 24), # 4个观测值
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2) # 2个动作
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr = 0.001)

    def forward(self, inputs):
        return self.fc(inputs)


env = gym.envs.make('CartPole-v1')
env = env.unwrapped
net = MyNet()  #最新网络Actor
net2 = MyNet() #延迟网络Critic，不用学习和反向传播

store_count = 0    # 记忆库行数/条数
store_size = 2000  # 记忆库大小
decline = 0.6      # 衰减系数
learn_time = 0     # 学习次数
update_time = 20   # 更新延迟网络的次数间距
gama = 0.9         # 90%衰减
b_size = 1000      # 一次从记忆库取batchSize条训练
store = np.zeros((store_size, 10))  # 记忆库 s, a, s_, r  4种观测状态 + 2种动作 + 4种观测状态 + 奖励
start_study = False
for i in range(50000):
    s = env.reset() #游戏重启
    while True:
        if random.randint(0,100) < 100*(decline**learn_time): # 首次0.6的0次方=1
            a = random.randint(0,1) # 一定概率选随机动作 0 或 1
        else:
            out = net(torch.Tensor(s)).detach()  # out中是[左走累计奖励r1, 右走累计奖励r2]
            a = torch.argmax(out).data.item()    # 选奖励高的动作    
        s_, r, done, _ = env.step(a)             # 执行动作，得到下状态

        # theta_threshold_radians杆子最大偏转角度 
        # s_[2]杆子偏转角度 abs(s_[2])杆子偏转角度(不管左右)
        # 偏转角=最大角，奖励=（最大角-偏转角）/最大角 = 0分。
        # 偏转角垂直=0， 奖励=（最大角-0）/最大角 = 1分。
        # 0.7控制奖励在-1,1之间
        # x_threshold偏移最大范围。 abs(s_[0])坐标偏移，中间为0
        # 偏转角(0,1)+坐标(0,1) = (0,2)    0.7和0.3让奖励落到(0,1)
        # r是立即奖励，网络负责预测奖励
        r = ( env.theta_threshold_radians - abs(s_[2]) ) / env.theta_threshold_radians * 0.7  +  ( env.x_threshold - abs(s_[0]) ) / env.x_threshold * 0.3
        
        # store_count % store_size 当条数满了就覆盖，没满就新增
        store[store_count % store_size][0:4] = s
        store[store_count % store_size][4:5] = a
        store[store_count % store_size][5:9] = s_
        store[store_count % store_size][9:10] = r
        store_count += 1
        s = s_

        if store_count > store_size:

            if learn_time % update_time == 0:
                net2.load_state_dict(net.state_dict())

            index = random.randint(0, store_size - b_size -1)
            b_s  = torch.Tensor(store[index:index + b_size, 0:4]) # index: index+b_size   取b_size行数据
            b_a  = torch.Tensor(store[index:index + b_size, 4:5]).long() #没学习前网络预测跟随机一样？
            b_s_ = torch.Tensor(store[index:index + b_size, 5:9])
            b_r  = torch.Tensor(store[index:index + b_size, 9:10])

            # 奖励=(1000,2) 动作b_a=(1000,1) gather用历史动作索引取历史动作奖励值，(1000,1)
            # 本状态 -> 预测 -> 本奖励(老动作取新奖励，不一定最大) -> 估计值
            q = net(b_s).gather(1, b_a) # 本状态算估计值 历史集合 q=r+q1这一步q r有了

            # detach不入计算图 max(n)=(最大值,最大值索引) max(1)找第1维非第0维的最大值 max(1)[0]第1维最大值的索引0 reshape(1000)->(1000,1)
            # 下状态 -> 预测o -> 下奖励(最大奖励) -> 下奖励+立即奖励 => 真实值
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1) 
            
            tq = b_r + gama * q_next  # 下状态算真实值 类似标签
            loss = net.mls(q, tq)     # 估计值 += (真实值 - 估计值) * 学习率 q和tq是公式左边右边
            net.opt.zero_grad()
            loss.backward()           # 网络根据真实值和估计值的损失，修正本奖励、本动作的预测，修正估计值
            net.opt.step()

            learn_time += 1
            if not start_study:
                print('start study')
                start_study = True
                break
        if done:
            break

        env.render()