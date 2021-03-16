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
        episode_frames = episode_frames.transpose(0, 1, 4, 2, 3) / 1.0
        # print(episode_frames.dtype)
        actions = actions[:, :, np.newaxis]

        self.episode_frames = torch.from_numpy(episode_frames.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))
        
        self.mean = torch.zeros_like(self.episode_frames[0])
        self.std = torch.zeros_like(self.episode_frames[0])
        
        self.mean[:, 0, :, :] = 182.6091
        self.mean[:, 1, :, :] = 182.6091
        self.mean[:, 2, :, :] = 182.6091

        self.std[:, 0, :, :] = 45.5565
        self.std[:, 1, :, :] = 47.6260
        self.std[:, 2, :, :] = 50.7284

    def __len__(self):
        return len(self.episode_frames)

    def __getitem__(self, idx):
        return {
            "episode_frames": (self.episode_frames[idx] - self.mean) / self.std,
            "actions": self.actions[idx]
        }
    
    def _calculate_mean_std(self):
        print(self.episode_frames.shape)
        std = torch.std(self.episode_frames, dim=(0, 1, 3, 4))
        mean = torch.mean(self.episode_frames, dim=(0, 1, 3, 4))
        print("mean: ", mean)
        print(mean.shape)
        print("std: ", std)
        print(std.shape)
        # mean:  tensor([182.6091, 182.6091, 182.6091])
        # torch.Size([3])
        # std:  tensor([45.5565, 47.6260, 50.7284])
        # torch.Size([3])


def postprocess(image):
    image_ = image.detach().clone()
    # print(image_.shape)
    mean = torch.ones_like(image_)
    std = torch.ones_like(image_)
    mean[:, 0, :, :] = 182.6091
    mean[:, 1, :, :] = 182.6091
    mean[:, 2, :, :] = 182.6091

    std[:, 0, :, :] = 45.5565
    std[:, 1, :, :] = 47.6260
    std[:, 2, :, :] = 50.7284

    image_ = image_ * std + mean
    image_ = torch.clamp(image_, min=0.0, max=255.0) / 255.
    return image_


if __name__ == "__main__":
    data_set = DMMDataset()
    data_set._calculate_mean_std()
