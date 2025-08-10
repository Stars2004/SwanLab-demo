import torch
import os
import torchvision
import swanlab
from torch.utils.data import DataLoader
from load_dataset import DatasetLoader
from torchvision.models import ResNet50_Weights


def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(TrainDataLoader), loss.item()))
        
        # 记录训练损失到 SwanLab
        swanlab.log({"train_loss": loss.item()})


def test(model, device, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print('Accuracy: {:.2f}%'.format(accuracy))
    
    # 记录测试准确率到 SwanLab
    swanlab.log({"test_acc": accuracy})


if __name__ == "__main__":
    # 超参数
    num_epochs = 20
    lr = 1e-4
    batch_size = 8
    num_classes = 2

    # 创建训练和验证数据集及其对应的数据加载器
    TrainDataset = DatasetLoader("./datasets/dataset_cats_and_dogs/train.csv")
    ValDataset = DatasetLoader("./datasets/dataset_cats_and_dogs/val.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

    # 加载预训练的 ResNet50 模型
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # 将全连接层的输出维度替换为 num_classes
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 初始化 SwanLab 配置参数
    swanlab.init(
        # 设置实验名
        experiment_name="ResNet50",
        # 设置实验介绍
        description="Train ResNet50 for cat and dog classification.",
        # 记录超参数
        config={
            "model": "resnet50",
            "optim": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_class": num_classes,
            "device": device,
        },
        # 设置 SwanLab 日志保存路径
        logdir="./swanlab_logs"  
    )    

    # 训练和测试
    for epoch in range(1, num_epochs + 1):
        train(model, device, TrainDataLoader, optimizer, criterion, epoch)
        if epoch % 4 == 0: 
            accuracy = test(model, device, ValDataLoader)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    torch.save(model.state_dict(), 'checkpoints/latest_checkpoint.pth')
    print("Training complete")
