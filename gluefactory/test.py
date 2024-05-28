import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Adjust the import path according to your project structure
from datasets.forest_images_dataset import ForestImagesDataset


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Example use
# def test_dataset_loading():
#     transformations = Compose([
#         Resize((256, 256)),  # Resize images to 256x256
#         ToTensor()           # Convert images to PyTorch tensors
#     ])
#     dataset = UnlabeledImageDataset(r'C:\Users\thoma\OneDrive\2023 Masters\Project\ProjectCode\external\glue-factory\gluefactory\datasets\forest_images', transform=transformations)
#     loader = DataLoader(dataset, batch_size=10, shuffle=True)
#
#     for images in loader:
#         print(images.shape)  # Should print [10, 3, 256, 256] for 10 images of size 256x256 with 3 color channels
#         break  # Only process the first batch


def test_dataset_loading():
    config = {'data_dir': r'C:/Users/thoma/OneDrive/2023 Masters/Project/ProjectCode/external/glue-factory/gluefactory/datasets/forest_images'}
    dataset = ForestImagesDataset(config['data_dir'])
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for images in loader:
        print(images.shape)
        break  # Just print the first batch to verify


if __name__ == '__main__':
    test_dataset_loading()
