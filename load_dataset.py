import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        self.csv_file = csv_path
        with open(self.csv_file, 'r') as file:
            self.data = list(csv.reader(file))

        # 项目文件夹路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def preprocess_image(self, image_path):
        full_path = os.path.join(self.current_dir, 'datasets', 'dataset_cats_and_dogs', image_path)
        image = Image.open(full_path)
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return image_transform(image)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label)

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    print(os.path.abspath(__file__))                      # f:\PyTorchProject\SwanLab-demo\load_dataset.py
    print(os.path.dirname(os.path.abspath(__file__)))   # f:\PyTorchProject\SwanLab-demo

    train_csv_path = './datasets/dataset_cats_and_dogs/train.csv'
    dataset = DatasetLoader(train_csv_path)
    print(dataset.data)  # [['train/cat/Sphynx_159_jpg.rf.022528b23ac690c34ad5d109c1782079.jpg', '0'], ...]

    print(dataset[0])
    print(dataset[0][0].shape, dataset[0][1])  # torch.Size([3, 256, 256]) 0
