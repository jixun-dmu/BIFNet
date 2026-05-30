import torch 
import torch.nn as nn 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)
    



class Attention(nn.Module):
    def __init__(self,channel=768):
        super(Attention,self).__init__()
        self.channel=channel
        self.CA=ChannelAttention(768)
        self.SA=SpatialAttention()

        self.CBR_t1=nn.Sequential(            
            nn.Conv2d(self.channel,self.channel,3,1,1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.CBR_r1=nn.Sequential(           
            #nn.Conv2d(2048,self.channel,3,1,1),
            nn.Conv2d(320,self.channel,3,1,1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.residual=nn.Sequential(                
            #nn.Conv2d(2048,self.channel,1,1),
            nn.Conv2d(320,self.channel,3,1,1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.layer1=nn.Sequential(                          
            nn.Conv2d(self.channel,self.channel,5,1,2),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.layer2=nn.Sequential(                          
            nn.Conv2d(self.channel,self.channel,3,1,1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )

        self.layer3=nn.Sequential(                         
            nn.Conv2d(self.channel,self.channel,1,1),
            nn.BatchNorm2d(self.channel),
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

    def forward(self,x1,x2):    
        out1=self.CBR_t1(x1)+self.CBR_r1(x2)
        out2=out1*self.SA(self.CBR_t1(x1))
        out3=out1*self.CA(self.CBR_r1(x2))
        out3=out2+x1+out3+self.residual(x2)
        out4=out3+self.layer1(out3)
        out5=out3+self.layer2(out4)
        out6=self.layer3(out5)+self.layer2(out4)+self.layer1(out3)
        #out=torch.sigmoid(out6)
        out=out6
        return out
