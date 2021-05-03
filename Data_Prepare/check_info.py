import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf
from torch.utils.data import DataLoader

# mean and std
class MyDataset(Dataset):
    def __init__(self, data_dir, data_id_list):
        self.img_dict, self.mask_dict = {},{}
        self.data_id_list = data_id_list
        self.data_dir = data_dir
        for case in data_id_list:
            self.img_dict[case] = case

    def __getitem__(self,idx):
        data_idx = self.data_id_list[idx]
        img = Image.open(os.path.join(self.data_dir, self.img_dict[data_idx])).resize((512, 512),Image.LANCZOS)
        img = ttf.to_tensor(img)
        return img.float()

    def __len__(self):
        return len(self.data_id_list)

data_dir = '/home/lingjia/Documents/tmp/0207_10331422510'
data_id_list = [name for name in os.listdir(data_dir) if not 'mask' in name and 'png' in name]
dataset = MyDataset(data_dir, data_id_list)
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print('mean')
print(mean)
print('std')
print(std)