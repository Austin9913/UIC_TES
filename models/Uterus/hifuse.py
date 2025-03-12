from models.Uterus.HiFuse_main.main_model import *
from models.Uterus.HiFuse_pretrain_main.HiFuse_pretrain_main.main_model import main_model as create_model

class hifuse(nn.Module):
    def __init__(self,num_classes):
        super(hifuse, self).__init__()
        model_path="/home/lvxinpeng/uterus/Uterus_Dis_Cl/models/pre/HiFuse_ImageNet1K.pth"
        # 加载.pth文件中的权重


        self.model = HiFuse_New(num_classes=num_classes)
        # state_dict = torch.load(model_path)
        # # 将权重加载到模型中
        # self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
def test_model():
    # 假设我们有一个分类任务，有10个类别
    num_classes = 10
    model = hifuse(num_classes)

    # 定义输入数据的大小，这里假设输入是一个3通道的224x224图像
    # 批量大小为1，即batch_size=1
    batch_size, channels, height, width = 1, 3, 224, 224
    input_data = torch.randn(batch_size, channels, height, width)

    # 将模型设置为评估模式
    model.eval()

    # 测试模型的前向传播
    with torch.no_grad():
        output = model(input_data)

    # 打印输出
    print(output)

# 定义主函数
def main():
    # 调用测试函数
    test_model()

    # 在这里可以添加更多的代码，例如保存模型，或者执行其他任务

# 当这个脚本被Python解释器直接执行时，main函数将被调用
if __name__ == '__main__':
    main()