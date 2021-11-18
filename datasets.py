from torch.utils.data.dataset import Dataset

class ImageTextPairDataset(Dataset):
    def __init__(self):
        
        pass
    def __getitem__(self, idx):


        image_tensor, text_prompt = [], []
        return image_tensor, text_prompt # tensor : 3 x 224 x 224, "happy cat"

    def __len__(self):
        pass
        # return 
