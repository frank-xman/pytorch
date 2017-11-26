import torch
data=[[2,2],[3,4]]
tensor=torch.FloatTensor(data)
#print  torch.mm(data,data)
print  torch.mm(tensor,tensor)
