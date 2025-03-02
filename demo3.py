#encoding=utf-8
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt
from torch.autograd import Variable
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=torch.autograd.Variable(x),Variable(y)
plt.scatter(x.data.numpy(),y.data.numpy())

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # 隐藏层线性输出
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # 隐藏层线性输出

        self.predict = torch.nn.Linear(n_hidden3, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden1(x))
	x = F.relu(self.hidden2(x))  
	x = F.relu(self.hidden3(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden1=10, n_hidden2=5, n_hidden3=5, n_output=1)

print(net)  # net 的结构
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.125)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
plt.ion()
plt.show()

for t in range(100000):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上i	
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
