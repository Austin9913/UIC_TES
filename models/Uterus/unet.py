import torch.nn as nn
import torch

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self, num_classes=512):

        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]

        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)


        #输出

        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.ReLU(),
            # BCELoss
        )
        self.newo = nn.Sequential(
            # nn.Flatten(2,-1),
            # nn.Flatten(0,1),
            # nn.Linear( 224 * 224, num_classes),
            nn.Flatten(1,-1),
            nn.Linear( 3*224 * 224, num_classes),
            nn.ReLU(),

        )
        self.conv_transpose_layer = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1d_layer = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=3)
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        out = self.newo(out)

        return out

#添加预训练权重
class unet(nn.Module):
    def __init__(self, num_classes=512):
        super(unet, self).__init__()
        self.model = UNet(num_classes)
        # ptrh_unet='/home/lvxinpeng/uterus//Uterus_Dis_Cl/models/pre/Unet.pt'
        # self.model.load_state_dict(torch.load(ptrh_unet),strict=False)
        # 加载模型
        checkpoint = torch.load('/home/lvxinpeng/uterus//Uterus_Dis_Cl/models/pre/Unet.pt')
        # 获取模型的当前状态字典
        model_state_dict =  self.model.state_dict()
        for key, value in checkpoint.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    # 执行复制操作
                    model_state_dict[key].copy_(value)
        # 迭代检查点的状态字典，只加载匹配的键
        # 加载模型的状态字典
        #print(model_state_dict)
        self.model.load_state_dict(model_state_dict)
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    test_data=torch.randn(4,3,224,224).cuda()
    net=unet().cuda()
   # print(net)
    # print(test_data)
    out=net(test_data)
    print(out.shape)