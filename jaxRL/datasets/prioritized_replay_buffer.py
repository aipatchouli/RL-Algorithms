import numpy as np
import random

prioritized_replay_epsilon = 0.01  #避免0误差导致永远不会被采样
prioritized_replay_max_error = 1.0  # 出于稳定性的考虑，将td_error裁剪

class ReplayMemory(object):

    def __init__(self, max_size, alpha, beta, beta_increment_per_sampling):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.size = 0
        self.idx = 0
        #建立了两个二叉树，一个的父节点存储子节点的优先级的和，另一个的父节点存储子节点优先级中的最小值
        self.priority_sum = [0 for _ in range(2 * self.max_size)]
        self.priority_min = [float('inf') for _ in range(2 * self.max_size)
                             ]  #float('inf')表示正无穷
        self.max_priority = 1
        #定义一个用于存放数据的字典，字典里面的每个值都是和经验池相等的列表，0初始化
        #索引从1开始，与二叉树对齐
        self.data = {
            'obs': np.zeros(shape=(max_size, 4), dtype=np.float32),
            'action': np.zeros(shape=max_size, dtype=np.int32),
            'reward': np.zeros(shape=max_size, dtype=np.float32),
            'next_obs': np.zeros(shape=(max_size, 4), dtype=np.float32),
            'done': np.zeros(shape=max_size, dtype=np.float32)
        }

    # 增加一条经验到经验池中
    def append(self, obs, action, reward, next_obs, done):

        self.data['obs'][self.idx] = obs  #存储数据
        self.data['action'][self.idx] = action
        self.data['reward'][self.idx] = reward
        self.data['next_obs'][self.idx] = next_obs
        self.data['done'][self.idx] = done

        self.size = min(self.max_size, self.idx + 1)  #获取当前的容量

        priority_alpha = self.max_priority**self.alpha  #样本第一次存储时赋予最大概率，保证有机会被抓取第二次

        self.set_priority_min(self.idx, priority_alpha)
        self.set_priority_sum(self.idx, priority_alpha)

        self.idx = self.idx + 1  #更新索引
        if self.idx >= self.max_size:  #样本被覆盖（丢弃）之前，索引一直保持不变
            self.idx = 0

    #每输入一条数据就从叶子节点开始遍历该叶子节点所属的二叉树分支一直到根部，并分别更新父节点
    #根据新数据更新节点，并更新最小优先级，
    def set_priority_min(self, idx, priority_alpha):
        idx += self.max_size  #后半部分存储样本，前面存储父节点
        self.priority_min[idx] = priority_alpha  #将优先级添加进二叉树
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(
                self.priority_min[2 * idx],
                self.priority_min[2 * idx + 1])  #用正无穷进行初始化，即使右节点无赋值
            #左节点无论值为多少，其父节点仍未左节点的值

    #根据新数据更新节点，并更新和
    def set_priority_sum(self, idx, priority):
        idx += self.max_size
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[
                2 * idx] + self.priority_sum[2 * idx + 1]

    #求和二叉树的根节点保存着所有叶节点的优先级总和
    def replay_priority_sum(self):
        return self.priority_sum[1]

    #最小值二叉树的根节点保存着所有叶节点的中的最小优先级
    def replay_priority_min(self):
        return self.priority_min[1]

    #从根部遍历二叉树，寻找指定范围的的叶子的索引
    def find_transition_idx(self, num):
        idx = 1  #根节点索引为1
        while idx < self.max_size:  #在整个树内遍历，流程如下
            if self.priority_sum[
                    idx * 2] > num:  #先与当前父节点的左子树节点比较，如果左子树节点比输入的数大，就进入这个左分支
                idx = 2 * idx  #因为对父节点来说，左子节点的索引是父节点索引的两倍，之后将左子节点作为父节点，从左子节点继续遍历
            else:  #进入右分支
                num -= self.priority_sum[idx * 2]  #同时减去该行左子树节点的值
                idx = 2 * idx + 1  #因为对父节点来说，右子节点的索引是父节点索引的两倍加1，之后将右子节点作为父节点，之后从右子节点继续遍历
        return idx - self.max_size  #样本的索引=优先级索引-容量

    # 从经验池中选取经验出来
    def sample(self, batch_size):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32),
            'obs': np.zeros(shape=(batch_size, 4), dtype=np.float32),
            'action': np.zeros(shape=batch_size, dtype=np.int32),
            'reward': np.zeros(shape=batch_size, dtype=np.float32),
            'next_obs': np.zeros(shape=(batch_size, 4), dtype=np.float32),
            'done': np.zeros(shape=batch_size, dtype=np.float32)
        }

        #按照概率从经验池中采样样本，获取样本的索引
        for i in range(batch_size):
            p = random.random() * self.replay_priority_sum(
            )  #将random.random()产生的随机数作为系数，获取值p
            idx = self.find_transition_idx(p)  #获取索引
            samples['indexes'][i] = idx

        #计算所有样本的可能最大权重系数，用于之后的归一化
        prob_min = self.replay_priority_min() / self.replay_priority_sum()
        max_weight = (prob_min * self.size)**(-self.beta)

        #获取抽取样本的权重系数并归一化
        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[
                idx + self.max_size] / self.replay_priority_sum()
            weight = (prob * self.size)**(-self.beta)
            samples['weights'][i] = weight / max_weight  #归一化

        self.beta = min(self.beta + self.beta_increment_per_sampling, 1.0)

        for k, v in self.data.items():  #返回可遍历的(键, 值) 元组数组
            samples[k] = v[samples['indexes']]
        return samples

    #更新优先级
    def update_priority(self, indexes, td_errors):
        td_errors += prioritized_replay_epsilon
        clipped_errors = np.minimum(td_errors.numpy(),
                                    prioritized_replay_max_error)
        for idx, td_error in zip(indexes, clipped_errors):
            priority_alpha = td_error**self.alpha
            self.set_priority_min(idx, priority_alpha)
            self.set_priority_sum(idx, priority_alpha)

    #输出队列的长度
    def __len__(self):
        return self.size