#encoding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

import torch.utils.data as Data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#encoding =utf-8
#hyper paramters
EPOCH=2
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.0126
DOWNLOAD_MNIST=False

train_data=torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),#(0,1) (0-255)
        download=DOWNLOAD_MNIST
        )
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]
class RNN(nn.Module):
	def __init__(self):
		super(RNN,self).__init__()
		self.rnn=nn.LSTM(
			input_size=28,
			hidden_size=64,
			num_layers=2,
			batch_first=True,
		)
		self.out=nn.Linear(64,10)
	def forward(self,x):
		r_out,(h_n,h_c)=self.rnn(x,None)
		out=self.out(r_out[:,-1,:])
		return out
		#([batch,tiime_step,input])
	


rnn=RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() 
for epoch in range(EPOCH):	
	for step,(x,y) in enumerate(train_loader):
 		b_x = Variable(x.view(-1, 28, 28))   # reshape x to (batch, time_step, input_size)
        	b_y = Variable(y)   # batch y

        	output = rnn(b_x)               # rnn output
        	loss = loss_func(output, b_y)   # cross entropy loss
        	optimizer.zero_grad()           # clear gradients for this training step
        	loss.backward()                 # backpropagation, compute gradients
        	optimizer.step()     
		if step%50==0:
			test_out=rnn(test_x.view(-1,28,28))
			pred_y=torch.max(test_out,1)[1].data.numpy().squeeze()
			accuracy = sum(pred_y == test_y) / float(test_y.size)
			print accuracy    
			#test_output = rnn(test_x[:10].view(-1, 28, 28))
			#pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
			print(pred_y[:40], 'prediction number')
			print(test_y[:40], 'real number')

