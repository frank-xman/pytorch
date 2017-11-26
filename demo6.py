import torch
from troch.autograd import Variable
torch.manual_seed(1)    # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
# x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) 
# noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
# 建网络
	net1 = torch.nn.Sequential(
        	torch.nn.Linear(1, 10),
      		torch.nn.ReLU()
		torch.nn.Linear(10, 1)
    )
    		optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    		loss_func = torch.nn.MSELoss()

    # 训练
    	for t in range(100):
        	prediction = net1(x)
        	loss = loss_func(prediction, y)
        	optimizer.zero_grad()
        	loss.backward()
        	
		optimizer.step()
