from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter

from pixyz.models import Model
from pixyz.losses import ARLoss, KullbackLeibler, NLL
from pixyz.distributions import Bernoulli, Normal


def init_dataset(f_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    data_dir = '../data'
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda data: data[0])
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=mnist_transform),
        batch_size=f_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=mnist_transform),
        batch_size=f_batch_size, shuffle=True, **kwargs)

    fixed_t_size = 28
    return train_loader, test_loader, fixed_t_size


if __name__ == '__main__':
    x_dim = 28
    h_dim = 100
    z_dim = 64
    t_max = x_dim

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    class Phi_x(nn.Module):
        def __init__(self):
            super(Phi_x, self).__init__()
            self.fc0 = nn.Linear(x_dim, h_dim)

        def forward(self, x):
            return F.relu(self.fc0(x))

    class Phi_z(nn.Module):
        def __init__(self):
            super(Phi_z, self).__init__()
            self.fc0 = nn.Linear(z_dim, h_dim)

        def forward(self, z):
            return F.relu(self.fc0(z))

    f_phi_x = Phi_x().to(device)
    f_phi_z = Phi_z().to(device)

    class Generator(Bernoulli):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z", "h_prev"], var=["x"])
            self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, h_dim)
            self.fc3 = nn.Linear(h_dim, x_dim)
            self.f_phi_z = f_phi_z

        def forward(self, z, h_prev):
            h = torch.cat((self.f_phi_z(z), h_prev), dim=-1)
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    class Prior(Normal):
        def __init__(self):
            super(Prior, self).__init__(cond_var=["h_prev"], var=["z"])
            self.fc1 = nn.Linear(h_dim, h_dim)
            self.fc21 = nn.Linear(h_dim, z_dim)
            self.fc22 = nn.Linear(h_dim, z_dim)

        def forward(self, h_prev):
            h = F.relu(self.fc1(h_prev))
            return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x", "h_prev"], var=["z"])
            self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
            self.fc21 = nn.Linear(h_dim, z_dim)
            self.fc22 = nn.Linear(h_dim, z_dim)
            self.f_phi_x = f_phi_x

        def forward(self, x, h_prev):
            h = torch.cat((self.f_phi_x(x), h_prev), dim=-1)
            h = F.relu(self.fc1(h))
            return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

    class Recurrence(nn.Module):
        def __init__(self):
            super(Recurrence, self).__init__()
            self.rnncell = nn.GRUCell(h_dim * 2, h_dim).to(device)
            self.f_phi_x = f_phi_x
            self.f_phi_z = f_phi_z
            self.hidden_size = self.rnncell.hidden_size

        def forward(self, x, z, h_prev):
            h_next = self.rnncell(torch.cat((self.f_phi_z(z), self.f_phi_x(x)), dim=-1), h_prev)
            return h_next

    prior = Prior().to(device)
    decoder = Generator().to(device)
    encoder = Inference().to(device)
    recurrence = Recurrence().to(device)

    # define the loss function

    def vrnn_step_fn(t, x, h_prev=None, h=None, z=None):
        z = encoder.sample({"x": x, "h_prev": h})["z"]
        h_next = recurrence(x, z, h)
        return {'x': x, 'h_prev': h, 'h': h_next, 'z': z}

    step_loss = (NLL(decoder) + KullbackLeibler(encoder, prior)).mean()
    loss = ARLoss(step_loss, last_loss=None,
                  step_fn=vrnn_step_fn, max_iter=t_max,
                  series_var=['x'], input_var=['x', 'h'])

    print(loss)
    vrnn = Model(loss, distributions=[encoder, decoder, prior, recurrence],
                 optimizer=optim.Adam, optimizer_params={'lr': 1e-3})

    def data_loop(epoch, loader, model, device, train_mode=False):
        mean_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            batch_size = data.size()[0]
            x = data.transpose(0, 1)
            h = torch.zeros(batch_size, recurrence.hidden_size).to(device)
            if train_mode:
                mean_loss += model.train({'x': x, 'h': h}).item() * batch_size
            else:
                mean_loss += model.test({'x': x, 'h': h}).item() * batch_size

        mean_loss /= len(loader.dataset)
        if train_mode:
            print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
        else:
            print('Test loss: {:.4f}'.format(mean_loss))
        return mean_loss

    train_loader, test_loader, t_max = init_dataset(32)

    def generation(batch_size):
        x = []
        h = torch.zeros(batch_size, recurrence.hidden_size).to(device)
        for step in range(t_max):
            z_t = prior.sample({'h_prev': h})['z']
            x_t = decoder.sample({'h_prev': h, 'z': z_t})['x']
            h = recurrence(x_t, z_t, h)
            x.append(x_t[None, :])
        x = torch.cat(x, dim=0).transpose(0, 1)
        return x

    writer = SummaryWriter()

    for epoch in range(100):
        train_loss = data_loop(epoch, train_loader, vrnn, device, train_mode=True)
        test_loss = data_loop(epoch, test_loader, vrnn, device)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)

        sample = generation(32)[:, None]
        writer.add_image('Image_from_latent', sample, epoch)
