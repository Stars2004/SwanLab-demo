import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision


# 加载与训练中使用的相同结构的模型
def load_model(checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model


# 加载图像并预处理
def process_image(image, image_size):
    # Define the same transforms as used during training
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocessing(image).unsqueeze(0)   # [1, C, H, W]
    return image


# 图像预测
def predict(image):
    classes = {'0': 'cat', '1': 'dog'}  # Update or extend this dictionary based on your actual classes
    image = process_image(image, 256)  # Using the image size from training
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).squeeze()  # Apply softmax to get probabilities
    # Mapping class labels to probabilities
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities


if __name__ == "__main__":
    # 加载模型权重
    checkpoint_path = './checkpoints/latest_checkpoint.pth'
    num_classes = 2
    model = load_model(checkpoint_path, num_classes)

    # 定义 Gradio Interface
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=num_classes),
        title="Cat vs Dog Classifier",
    )

    # 启动 Gradio 应用
    # iface.launch()