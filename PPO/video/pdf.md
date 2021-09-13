Actor critic方法对干扰很敏感  
限制对策略网络的更新
根据新策略与旧策略的比率进行更新 
clip loss函数，取最小值为下界  
追踪记忆的固定长度轨迹
每个数据样本使用多个网络更新
小批量随机梯度上升

Memory indices = [0, 1, 2, …, 19]
Batches从batch_size的倍数开始[0,5,10,15]
Shuffle memories 然后取出 batch size

两个不同的网络而不是共享的输入
Critic 评估状态(不是一对s a)
Actor根据当前状态决定要做什么
网络输出一个分布的probs (softmax)

Memory长度固定的T(比如20步)
Shuffle memories and sample batches (5) 
追踪状态，行动，奖励，完成，价值，记录概率 
    states, actions, rewards, dones, values, log probs 
对每batch 执行4个epochs 的更新

为clip/min操作定义ε (~0.2)
优势函数At的Lambda是一个平滑参数(~0.95)

回报=优势+critic值(来自mem)
Lcritic = MSE(回报-critic值(来自网络))

总损失 = clip_actor + critic