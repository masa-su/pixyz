from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def imshow(img_tensors):
    img = torchvision.utils.make_grid(img_tensors)
    npimg = img.numpy()
    plt.figure(figsize=(16, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class DMMDataset(Dataset):
    def __init__(self, pickle_path="cartpole_28.pickle"):
        
        with open(pickle_path, mode='rb') as f:
            data = pickle.load(f)
        episode_frames, actions = data
        # episode_frames: np.array([episode_num, one_episode_length, height, width, Channels]) (10000, 30, 28, 28, 3)
        # actions:             np.array([episode_num, one_episode_length]) (10000, 30)
        # HWC → CHW
        episode_frames = episode_frames.transpose(0, 1, 4, 2, 3)
        actions = actions[:, :, np.newaxis]
        
        self.episode_frames = torch.from_numpy(episode_frames.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))
            
    def __len__(self):
        return len(self.episode_frames)
 
    def __getitem__(self, idx):
        return {
            "episode_frames": self.episode_frames[idx] / 255,
            "actions": self.actions[idx]
        }


if __name__ == "__main__":
    pass
