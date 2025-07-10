import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p=8
group_angle = [2*k*np.pi/p+np.pi/8 for k in range(p)]
tran_to_partial_coef_0 = [torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,np.cos(x),np.sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,-np.sin(x),np.cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(np.cos(x),2),2*np.cos(x)*np.sin(x),pow(np.sin(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,-np.cos(x)*np.sin(x),pow(np.cos(x),2)-pow(np.sin(x),2),np.sin(x)*np.cos(x),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(np.sin(x),2),-2*np.cos(x)*np.sin(x),pow(np.cos(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,-pow(np.cos(x),2)*np.sin(x),pow(np.cos(x),3)-2*np.cos(x)*pow(np.sin(x),2),-pow(np.sin(x),3)+2*pow(np.cos(x),2)*np.sin(x), pow(np.sin(x),2)*np.cos(x),0,0,0,0,0],
                                     [0,0,0,0,0,0,np.cos(x)*pow(np.sin(x),2),-2*pow(np.cos(x),2)*np.sin(x)+pow(np.sin(x),3),pow(np.cos(x),3)-2*np.cos(x)*pow(np.sin(x),2),np.sin(x)*pow(np.cos(x),2),0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0,0,pow(np.sin(x),2)*pow(np.cos(x),2),-2*pow(np.cos(x),3)*np.sin(x)+2*np.cos(x)*pow(np.sin(x),3),pow(np.cos(x),4)-4*pow(np.cos(x),2)*pow(np.sin(x),2)+pow(np.sin(x),4),-2*np.cos(x)*pow(np.sin(x),3)+2*pow(np.cos(x),3)*np.sin(x),pow(np.sin(x),2)*pow(np.cos(x),2)]], dtype = torch.float32, device = device) for x in group_angle]





partial_dict_0 = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
                    [[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
                    [[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
                    [[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]], dtype = torch.float32, device=device)


def get_coef(weight,num_inputs,num_outputs):
        #weight.size 1,7,3,3 or 56,7,3,3
        
        transformation = partial_dict_0[[0,1,2,3,4,5,7,8,12],1:4,1:4] #9*3*3
        transformation = transformation.view([9,9])
        transformation = transformation.to(device)
        #print('transformation',transformation.device)
        inv_transformation = transformation.inverse()#inverse matrix
        
        betas = torch.reshape(weight,(-1,9))#56*7*9
        betas = betas.to(device)
        betas = torch.mm(betas,inv_transformation)# 56*7*9
        betas = torch.reshape(betas,(num_inputs,num_outputs,9))
        
        #print('betas',betas.device)
        return betas

def z2_kernel(weight,num_inputs,num_outputs,p,partial,tran, device):
    og_coef = torch.reshape(weight,(num_inputs*num_outputs,9)) #(56*7)*9
    #print('og',og_coef.type())
    tran = [torch.tensor(a, dtype=torch.float32, device=device) for a in tran]
    partial_coef = [torch.mm(og_coef,a) for a in tran]#8,(56*7)*15
    partial = torch.reshape(partial,(15,25))#15*25
    partial=partial.to(device)
    
    kernel = [torch.mm(a,partial) for a in partial_coef]#8,(56*7)*25
    kernel = torch.stack(kernel,dim=1)#(56*7)*8*25
    kernel = torch.reshape(kernel,(num_outputs*p,num_inputs,5,5))#56*56*5*5 or 56*1*5*5
    kernel=kernel.to(device )
    #print('z2kernel',kernel.device)
    return kernel

class open_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran
        
        self.weight = nn.Parameter(torch.empty(self.num_inputs,self.num_outputs,3,3, device=device))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):
        # device = input.device
        betas=get_coef(self.weight,self.num_inputs,self.num_outputs)
        
        kernel=z2_kernel(betas,self.num_inputs,self.num_outputs,self.p,self.partial,self.tran, device)
        
         

        input_shape = input.size()#input_size: 128,1,h,w & 128,56,h,w
        input = input.view(input_shape[0], self.num_inputs, input_shape[-2], input_shape[-1])
        
        #y_size: 128,56,h,w
        
        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding=1)

        batch_size, _, ny_out, nx_out = outputs.size()
        outputs = outputs.view(batch_size, self.num_outputs*self.p, ny_out, nx_out)
        #y_size: 128,7*8,h,w


        return outputs

class g_bn(nn.Module): # Batch Normalization Layer
    def __init__(self,p=8):
        super(g_bn, self).__init__()
        self.p=p
        self.bn=nn.BatchNorm2d(32)

    def forward(self, inputs):
        # inputs = inputs.to(device)
        channel,height,width = list(inputs.size())[1:]
        inputs = inputs.view(-1,int(channel/p),p,height,width)
        inputs = inputs.view(-1,int(channel/p),height*p,width)
        
        outputs=self.bn(inputs)
        
        outputs=outputs.view(-1,int(channel/p),p,height,width,)
        outputs = outputs.view(-1,channel,height,width)
        

        return outputs

class g_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        if num_inputs % p != 0:
            raise ValueError(f"num_inputs={num_inputs} must be divisible by p={p}")
        self.num_inputs = num_inputs // p

        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran
        
        self.weight = nn.Parameter(torch.empty(self.p*self.num_inputs,self.num_outputs,3,3, device=device))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):
        
        #print(self.weight.size())
        betas=get_coef(self.weight,self.num_inputs*self.p,self.num_outputs)
        og_coef = betas.view(self.num_inputs*self.p*self.num_outputs,9)#(56*7)，9
        
        tran_to_partial_coef = self.tran #8，9*15
        partial_coef = [torch.mm(og_coef,a) for a in tran_to_partial_coef] #8，（56*7）*15

        
        partial_dict = self.partial
        partial_dict = partial_dict.view(15,25)#15*25
        partial_dict = partial_dict.to(device)
        

        og_kernel_list = [torch.mm(a,partial_dict) for a in partial_coef] #8，（56*7）*25
        og_kernel_list = [og_kernel.view(self.num_inputs,self.p,self.num_outputs,25) for og_kernel in og_kernel_list] #8，（7*8*7*25）
        
        #print(og_kernel_list[0][:,0:,:].size(),og_kernel_list[0][:,:0,:].size())
        og_kernel_list = [torch.cat([og_kernel_list[k][:,-k:,:],og_kernel_list[k][:,:-k,:]],dim=1) for k in range(p)] #8，（7*8*7*25）
        
        
        kernel = torch.stack(og_kernel_list,dim=3)#7,8,7,8,25
        kernel = kernel.view(self.num_outputs * self.p, self.num_inputs * self.p, 5, 5)

        # kernel = kernel.view(self.num_inputs*self.p,self.num_outputs*self.p,5,5)#56,56,5,5
          
        
        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding=1)
        batch_size, _, ny_out, nx_out = outputs.size()
        expected_size = batch_size * self.num_outputs * self.p * ny_out * nx_out
        actual_size = outputs.numel()
        # print("DEBUG:")
        # print("  input.shape:", input.shape)
        # print("  kernel.shape:", kernel.shape)
        # print("  outputs.shape (before reshape):", outputs.shape)
        # print("  batch_size:", batch_size)
        # print("  self.num_outputs:", self.num_outputs)
        # print("  self.p:", self.p)
        # print("  ny_out:", ny_out)
        # print("  nx_out:", nx_out)
        # print("  expected:", batch_size * self.num_outputs * self.p * ny_out * nx_out)
        # print("  actual:", outputs.numel())


        assert actual_size == expected_size, f"Shape mismatch before reshape: expected {expected_size}, got {actual_size}"

        outputs = outputs.reshape(batch_size, self.num_outputs * self.p, ny_out, nx_out)

        #y_size: 128,7*8,h,w
        

        return outputs

class P8_PDO_Conv_Z2(open_conv2d):

    def __init__(self, *args, **kwargs):
        super(P8_PDO_Conv_Z2, self).__init__(num_inputs=1, num_outputs=32  ,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)


class P8_PDO_Conv_P8(g_conv2d):

    def __init__(self, num_inputs, num_outputs, p=8):

        super(P8_PDO_Conv_P8, self).__init__(num_inputs=num_inputs, num_outputs= num_outputs,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)

class BN_P8(g_bn):
    
    def __init__(self, *args, **kwargs):
        super(BN_P8, self).__init__(p=8)

class PDO_eConvs(nn.Module):
    def __init__(self):
        super(PDO_eConvs, self).__init__()
        self.conv1 = P8_PDO_Conv_Z2(1,32,8)
        self.conv2 = P8_PDO_Conv_P8(256,32,8)
        self.conv3 = P8_PDO_Conv_P8(256,32,8)
        self.conv4 = P8_PDO_Conv_P8(256,32,8)
        self.conv5 = P8_PDO_Conv_P8(256,32,8)
        self.conv6 = P8_PDO_Conv_P8(256,32,8)
        self.dropout=nn.Dropout(p=0.19)
        self.bn1 = BN_P8(8)
        self.bn2 = BN_P8(8)
        self.bn3 = BN_P8(8)
        self.bn4 = BN_P8(8)
        self.bn5 = BN_P8(8)
        self.bn6 = BN_P8(8)
        self.maxpool2= nn.MaxPool2d(kernel_size=2, stride=2)
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(4 * 4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 10)

        
    def forward(self, x):

        x = self.dropout(self.bn1(F.relu(self.conv1(x))))
        #print(x.size())
        x = self.maxpool2(self.bn2(F.relu(self.conv2(x))))
        
        x = self.dropout(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout(self.bn5(F.relu(self.conv5(x))))
        x = self.dropout(self.bn6(F.relu(self.conv6(x))))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y=torch.nn.functional.log_softmax(x, dim=1)
        
        return y

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)