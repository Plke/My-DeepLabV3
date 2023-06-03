from torch.utils.data import DataLoader,Dataset
from PIL import Image
class MyDataset(Dataset):
    def __init__(self, root=r'data',txt_file=r'data', transforms=None):
        f=open(txt_file)

        self.img_names = f.readlines() 
        
        f.close()
        self.transforms=transforms
        self.root=root

    def __getitem__(self, index):
        
        img=Image.open('{}/imgs/{}.jpg'.format(self.root,self.img_names[index][12:23]))
        mask=Image.open('{}/labels/{}.png'.format(self.root,self.img_names[index][12:23]))

        
        if self.transforms is not None:
            img,mask= self.transforms(img,mask)
    
        return img,mask

    def __len__(self):
        return len(self.img_names)