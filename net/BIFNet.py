import torch 
import torch.nn as nn 
from MobileNetV2 import *
from rest_new import ResT as Pre_Rest
from decoder import *
from Attention import *
import math
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_one(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_one(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self.with_pos:
            x = self.pos(x)
        return x


class ResT_my(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, apply_transform=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        # classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
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

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x1 = x

        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x2 = x

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x3 = x

        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        # x = self.avg_pool(x).flatten(1)
        # x = self.head(x)
        return x1, x2, x3, x


class residual1(nn.Module):                             
    def __init__(self):
        super(residual1,self).__init__()
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
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
        out=self.up(x)
        return out 
    
class residual2(nn.Module):                                 
    def __init__(self):
        super(residual2,self).__init__()
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer=nn.Sequential(
            nn.Conv2d(768,384,1,1),
            nn.BatchNorm2d(384),
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
        out=self.up(x)
        out=self.layer(out)
        out=self.up(out)
        return out 
    

class residual3(nn.Module):                                
    def __init__(self,channel=768):
        super(residual3,self).__init__()
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer1=nn.Sequential(
            nn.Conv2d(channel,channel//2,1,1),
            nn.BatchNorm2d(channel//2),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(channel//2,channel//4,1,1),
            nn.BatchNorm2d(channel//4),
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
        out=self.up(x)
        out=self.layer1(out)
        out=self.up(out)
        out=self.layer2(out)
        out=self.up(out)
        return out 
    

class up(nn.Module):                                  
    def __init__(self,inchannel,outchannel):
        super(up,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,1,1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
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
        out=self.up(self.layer(x))
        return out






class BIFNet(nn.Module):
    def __init__(self):
        super(BIFNet,self).__init__()
        self.attention=Attention()               

        self.residual1=residual1()               
        self.residual2=residual2()               
        self.residual3=residual3()              
        self.decoder1=decoder(320,768,768)       
        self.decoder2=decoder(96,768,384,scale=4)       
        self.decoder3=decoder(32,384,192,scale=2)        
        self.decoder4=decoder(24,192,96,scale=1)       

        self.tu_1=up(768,384)                  
        self.tu_2=up(384,192)                 
        self.tu_3=up(192,96)                 

        self.mu_1=up(320,96)                  
        self.mu_2=up(96,32)                
        self.mu_3=up(32,24)                 


        self.MobileNetV2=mobilenet_v2(pretrained=True)           ##MobileNetV2
        self.ResT=ResT_my(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                            qkv_bias=True,
                            depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True)
        rest=Pre_Rest(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                            qkv_bias=True,
                            depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True) 
        rest.load_state_dict(torch.load('TCmodule/rest_large.pth'))           ##Rest
        pretrained_dict = rest.state_dict()
        model_dict = self.ResT.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.ResT.load_state_dict(model_dict)
        print('load transformer weight successfully')

        self.head=nn.Sequential(                             
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )



        self.layer1=nn.Conv2d(768, 1, kernel_size=3, stride=1, padding=1)
        self.layer2=nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1)
        self.layer3=nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)
        self.layer4=nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1)


    def forward(self,x):       

        t1,t2,t3,t4=self.ResT(x)                     
        m0,m1,m2,m3,m4=self.MobileNetV2(x)

        out_attention=self.attention(t4,m4)    

        d1_out=self.decoder1(m4,out_attention,t4)     

        d2_out=self.decoder2((m3+self.mu_1(m4)),(abs(self.residual1(out_attention)-d1_out)),(t3+self.tu_1(t4)))     

        d3_out=self.decoder3((m2+self.mu_2(m3)),(abs(self.residual2(out_attention)-d2_out)),(t2+self.tu_2(t3)))   

        d4_out=self.decoder4((m1+self.mu_3(m2)),(abs(self.residual3(out_attention)-d3_out)),(t1+self.tu_3(t2)))   

        out=self.head(d4_out)


        d1_out_1=self.layer1(d1_out)
        d2_out_1=self.layer2(d2_out) 
        d3_out_1=self.layer3(d3_out)
        d4_out_1=self.layer4(d4_out)



        #return torch.sigmoid(out)

        return out,d1_out_1,d2_out_1,d3_out_1,d4_out_1
    
