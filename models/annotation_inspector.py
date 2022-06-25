import torch
import torch.nn as nn

import timm

class conv_Block(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super(conv_Block,self).__init__()
        self.selu=nn.SELU()
        self.conv=nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchNorm=nn.BatchNorm2d(out_channels)
  
    def forward(self,x):
        x=self.conv(x)
        x=self.batchNorm(x)
        x=self.selu(x)
        return x
    
class Limb_Injector(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Limb_Injector,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.softmax=nn.Softmax(dim=-1)
        self.CB=conv_Block(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=1,stride=1, padding=0)
  
    def forward(self,feature,heat):
        heat_2=self.softmax(heat.view(heat.shape[0],heat.shape[1],-1)).view(heat.shape)
        li=[feature*(heat[:,i,:,:].unsqueeze(1)) for i in range(heat.shape[1])]
        cat=torch.cat(li,dim=1)
        out=torch.cat([cat,heat],dim=1)
        out=self.CB(out)
        return out

class Model_Inspector(nn.Module):
    
    def __init__(self,model_name,pretrained=False):
        
        super(Model_Inspector,self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.conv1=self.model.conv1
        self.bn1=self.model.bn1
        self.act1=self.model.act1
        self.maxpool=self.model.maxpool
        self.layer1=self.model.layer1
        
        self.limb_injection=Limb_Injector(256*14+14,256)
        
        self.layer2=self.model.layer2
        self.layer3=self.model.layer3
        self.layer4=self.model.layer4
        self.global_pool=self.model.global_pool
        self.fc=self.model.fc
        self.linear=nn.Linear(1000,1)
        self.sigmoid=nn.Sigmoid()
   
    def forward(self, x,heat):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        y=self.limb_injection(x,heat)
        y=self.layer2(y)
        y=self.layer3(y)
        y=self.layer4(y)
        y=self.global_pool(y)
        y=self.fc(y)
        y=self.linear(y)
        y=self.sigmoid(y)
        return y
        
    def save(self,optim,epoch):
        self.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': 0,
            }, './Chai{}.pth'.format(epoch))
    
    def load(self,optim,path):
        #checkpoint = torch.load(path,map_location=torch.device('cpu'))
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        


mx = Model_Inspector('resnet50d')
mx(torch.zeros(8, 3, 256, 256),torch.zeros(8, 14, 64, 64)).shape

c_model=Model_Inspector('resnet101d')

b_loss=nn.BCELoss()
c_optimiser = torch.optim.AdamW(c_model.parameters(), lr=0.001, betas=(0.5, 0.999))




# c_model.load(c_optimiser,'pretrained_weights/resnet_101d_Inspector_IR.pth')
print('inspector')