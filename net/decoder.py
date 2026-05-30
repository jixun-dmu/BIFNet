import torch
import torch.nn as nn 
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Split(nn.Module):
    def __init__(self,inchannel):
        super(Split,self).__init__()
        self.channel=inchannel//4
        self.split_indexes =(self.channel,self.channel,self.channel,self.channel)

        self.conv_1=nn.Sequential(
            nn.Conv2d(self.channel,self.channel,3,1,1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.conv_2=nn.Sequential(
            nn.Conv2d(self.channel,self.channel,3,1,2,dilation=2),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.conv_3=nn.Sequential(
            nn.Conv2d(self.channel,self.channel,5,1,2),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.layer=nn.Sequential(
            nn.Conv2d(inchannel,inchannel,3,1,1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )

        self.apply(self._init_weights)  
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x_id,x_conv_1,x_conv_2,x_conv_3=torch.split(x,self.split_indexes,dim=1)
        x_conv_1=self.conv_1(x_conv_1)
        x_conv_2=self.conv_2(x_conv_2)
        x_conv_3=self.conv_3(x_conv_3)
        out=self.layer(torch.cat([x_id,x_conv_1,x_conv_2,x_conv_3],dim=1))
        return out
    


class decoder(nn.Module):
    def __init__(self,channel1,channel2,channel3,scale=8): 
        super(decoder,self).__init__()
        self.channel=96
        self.Split=Split(self.channel*scale)
        self.up= nn.UpsamplingBilinear2d(scale_factor=2)

        self.layer1=nn.Sequential(     
            nn.Conv2d(channel1,self.channel*scale,3,1,1),   
            nn.BatchNorm2d(self.channel*scale),
            nn.ReLU()
        )

        self.layer2=nn.Sequential(     
            nn.Conv2d(channel2,self.channel*scale,3,1,1),   
            nn.BatchNorm2d(self.channel*scale),
            nn.ReLU()
        )

        self.layer3=nn.Sequential(    
            nn.Conv2d(channel3,self.channel*scale,3,1,1),   
            nn.BatchNorm2d(self.channel*scale),
            nn.ReLU()
        )

        self.layer4=nn.Sequential(    
            nn.Conv2d(self.channel*scale*3,self.channel*scale,3,1,1),   
            nn.BatchNorm2d(self.channel*scale),
            nn.ReLU()
        )

        self.residual=nn.Sequential(                
            nn.Conv2d(channel1,self.channel*scale,1,1),
            nn.BatchNorm2d(self.channel*scale),
            nn.ReLU()
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self,x1,x2,x3):  
        out1=self.layer1(x1)
        out2=self.layer2(x2)
        out3=self.layer3(x3)
        out4=self.Split(out1)+out2+self.Split(out3)
        out6=self.residual(x1)+self.Split(out1)
        out7=x3+self.Split(out3)
        out5=torch.cat([out4,out6,out7],dim=1)
        out=self.layer4(out5)
        out=self.up(out)
        return out

