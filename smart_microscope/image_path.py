import torch
from PIL import Image


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        #print("Dataframe: ", dataframe)
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = {'B': 0, 'M': 1} 
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label_str = self.dataframe.iloc[idx]["label"]
        
        # Convert to integer
        label = self.label_map[label_str] 
        
        # Load image on-the-fly
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        
        return image, label