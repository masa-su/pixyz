import torch
from torch import optim

from pixyz.models import Model
from pixyz.losses import AutoRegressiveSeriesLoss, KullbackLeibler, NLL


def create_vrnn(encoder, decoder, prior, f_phi_x, f_phi_z, rnncell, t_max,
                optimizer=optim.Adam, optimizer_params={'lr': 1e-3}):

    def vrnn_step_fn(t, x, h=None, h_prev=None, phi_x=None, phi_z=None, z=None):
        if t == 0:
            batch_size = x.size()[0]
            h = torch.zeros(batch_size, rnncell.hidden_size).to(x.device)
        h_prev = h
        z = prior.sample({"h_prev": h})["z"]
        phi_z = f_phi_z(z)
        phi_x = f_phi_x(x)
        h = rnncell(torch.cat((phi_x, phi_z), dim=-1), h)
        return {'x': x, 'h': h, 'h_prev': h_prev, 'phi_x': phi_x, 'phi_z': phi_z, 'z': z}

    step_loss = (NLL(decoder) + KullbackLeibler(encoder, prior)).mean()
    loss = AutoRegressiveSeriesLoss(step_loss, last_loss=None,
                                    step_fn=vrnn_step_fn, max_iter=t_max,
                                    series_var=['x'], input_var=['x'])

    print(loss)
    return Model(loss, distributions=[encoder, decoder, prior, f_phi_x, f_phi_z, rnncell],
                 optimizer=optimizer, optimizer_params=optimizer_params)


if __name__ == '__main__':
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, datasets
    from torchvision.utils import make_grid, save_image
    from tqdm import tqdm
    from pixyz.distributions import Bernoulli, Normal
    x_dim = 28
    h_dim = 100
    z_dim = 64
    t_max = x_dim
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    class Generator(Bernoulli):

        def __init__(self):
            super(Generator, self).__init__(cond_var=["phi_z", "h_prev"], var=["x"])

            self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, h_dim)
            self.fc3 = nn.Linear(h_dim, x_dim)

        def forward(self, phi_z, h_prev):
            h = torch.cat((phi_z, h_prev), dim=-1)
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
            super(Inference, self).__init__(cond_var=["phi_x", "h_prev"], var=["z"])

            self.fc1 = nn.Linear(h_dim + h_dim, h_dim)
            self.fc21 = nn.Linear(h_dim, z_dim)
            self.fc22 = nn.Linear(h_dim, z_dim)

        def forward(self, phi_x, h_prev):
            h = torch.cat((phi_x, h_prev), dim=-1)
            h = F.relu(self.fc1(h))
            return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

    class phi_x(nn.Module):
        def __init__(self):
            super(phi_x, self).__init__()

            self.fc0 = nn.Linear(x_dim, h_dim)

        def forward(self, x):
            return F.relu(self.fc0(x))

    class phi_z(nn.Module):
        def __init__(self):
            super(phi_z, self).__init__()

            self.fc0 = nn.Linear(z_dim, h_dim)

        def forward(self, z):
            return F.relu(self.fc0(z))

    prior = Prior().to(device)
    decoder = Generator().to(device)
    encoder = Inference().to(device)
    phi_x = phi_x().to(device)
    phi_z = phi_z().to(device)
    rnncell = nn.GRUCell(h_dim * 2, h_dim).to(device)
    vrnn = create_vrnn(encoder, decoder, prior, phi_x, phi_z, rnncell, t_max)

    def data_loop(epoch, loader, model, device, train_mode=False):
        mean_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            if train_mode:
                mean_loss += model.train({'x': data.transpose(0, 1)}) * data.size()[0]
            else:
                mean_loss += model.test({'x': data.transpose(0, 1)}) * data.size()[0]

        mean_loss /= len(loader.dataset)
        if train_mode:
            print('Epoch: {} Train loss: {:.4f}'.format(epoch, -mean_loss))
        else:
            print('Test loss: {:.4f}'.format(-mean_loss))
        return mean_loss

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
        # # lengthsの仕様と意義を理解する
        fixed_t_size = 28
        return train_loader, test_loader, fixed_t_size

    train_loader, test_loader, t_max = init_dataset(32)

    def generate(batch_size):
        x = []
        h = torch.zeros(batch_size, rnncell.hidden_size).to(device)
        for step in range(t_max):
            h_prev = h
            z_t = prior.sample({'h_prev': h})['z']
            phi_z_t = phi_z(z_t)
            x_t = decoder.sample({'h_prev': h_prev, 'phi_z': phi_z_t})['x']
            x.append(x_t[None, :])
            phi_x_t = phi_x(x_t)
            h = rnncell(torch.cat((phi_x_t, phi_z_t), dim=-1), h_prev)
        x = torch.cat(x, dim=0).transpose(0, 1)
        return x

    for epoch in range(10):
        train_loss = data_loop(epoch, train_loader, vrnn, device, train_mode=True)
        test_loss = data_loop(epoch, test_loader, vrnn, device)
        filename = '../data/vrnn_generate_%d.png' % epoch
        save_image(make_grid(generate(32)[:, None]), filename)
