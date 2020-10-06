# flake8: noqa: F841
from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from pixyz.distributions import Deterministic
from pixyz.models import GAN
from pixyz.distributions import InverseTransformedDistribution
from pixyz.flows import AffineCoupling, FlowList, Squeeze, Unsqueeze, Preprocess, ActNorm2d, ChannelConv
from pixyz.layers import ResNet
from pixyz.models import ML
from pixyz.distributions.mixture_distributions import MixtureModel
from pixyz.models import VI
from pixyz.utils import get_dict_values
from pixyz.distributions import Normal, Bernoulli, Categorical, ProductOfNormal
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
from pixyz.utils import print_latex

seed = 1
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

batch_size = 2
epochs = 2

# TODO: ファイル出力のコメントアウト
# TODO: モックデータのインプット
# TODO: 外部ライブラリ依存のコメントアウト


mock_mnist = [(torch.zeros(28 * 28), 0), (torch.ones(28 * 28), 1)]
mock_mnist_targets = torch.tensor([0, 1])
mock_cifar10 = [(torch.ones(3, 32, 32), 3), (torch.ones(3, 32, 32), 1)]


# # Conditional variational autoencoder (using the VAE class)
def test_run_cvae():
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    #
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:
    # In[4]:

    x_dim = 784
    y_dim = 10
    z_dim = 64

    # inference model q(z|x,y)
    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x", "y"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim + y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x, y):
            h = F.relu(self.fc1(torch.cat([x, y], 1)))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z,y)
    class Generator(Bernoulli):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z", "y"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim + y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z, y):
            h = F.relu(self.fc1(torch.cat([z, y], 1)))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # In[5]:

    print(prior)
    print_latex(prior)

    # In[6]:

    print(p)
    print_latex(p)

    # In[7]:

    print(q)
    print_latex(q)

    # In[8]:

    kl = KullbackLeibler(q, prior)
    print(kl)
    print_latex(kl)

    # In[9]:

    model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            z = q.sample({"x": x, "y": y}, return_all=False)
            z.update({"y": y})
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            recon = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return recon

    def plot_image_from_latent(z, y):
        with torch.no_grad():
            sample = p.sample_mean({"z": z, "y": y}).view(-1, 1, 28, 28).cpu()
            return sample

    def plot_reconstrunction_changing_y(x, y):
        y_change = torch.eye(10)[range(7)].to(device)
        batch_dummy = torch.ones(x.size(0))[:, None].to(device)
        recon_all = []

        with torch.no_grad():
            for _y in y_change:
                z = q.sample({"x": x, "y": y}, return_all=False)
                z.update({"y": batch_dummy * _y[None, :]})
                recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)
                recon_all.append(recon_batch)

            recon_changing_y = torch.cat(recon_all)
            recon_changing_y = torch.cat([x.view(-1, 1, 28, 28), recon_changing_y]).cpu()
            return recon_changing_y

    # In[13]:

    # writer = SummaryWriter()

    plot_number = 1

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    y_sample = torch.eye(10)[[plot_number] * 64].to(device)

    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_latent(z_sample, y_sample)
        recon_changing_y = plot_reconstrunction_changing_y(_x[:8], _y[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_change_y', recon_changing_y, epoch)
    # 
    # writer.close()

    # In[ ]:


# # Examples of creating and operating distributions in Pixyz
def test_run_distributions():
    # In[1]:
    # In[2]:
    # In[3]:

    x_dim = 20
    y_dim = 30
    z_dim = 40
    a_dim = 50
    batch_n = 2

    class P1(Normal):
        def __init__(self):
            super(P1, self).__init__(cond_var=["y", "a"], var=["x"], name="p_{1}")

            self.fc1 = nn.Linear(y_dim, 10)
            self.fc2 = nn.Linear(a_dim, 10)
            self.fc21 = nn.Linear(10 + 10, 20)
            self.fc22 = nn.Linear(10 + 10, 20)

        def forward(self, a, y):
            h1 = F.relu(self.fc1(y))
            h2 = F.relu(self.fc2(a))
            h12 = torch.cat([h1, h2], 1)
            return {"loc": self.fc21(h12), "scale": F.softplus(self.fc22(h12))}

    class P2(Normal):
        def __init__(self):
            super(P2, self).__init__(cond_var=["x", "y"], var=["z"], name="p_{2}")

            self.fc3 = nn.Linear(x_dim, 30)
            self.fc4 = nn.Linear(30 + y_dim, 400)
            self.fc51 = nn.Linear(400, 20)
            self.fc52 = nn.Linear(400, 20)

        def forward(self, x, y):
            h3 = F.relu(self.fc3(x))
            h4 = F.relu(self.fc4(torch.cat([h3, y], 1)))
            return {"loc": self.fc51(h4), "scale": F.softplus(self.fc52(h4))}

    p4 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["a"], features_shape=[a_dim], name="p_{4}")
    p6 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["y"], features_shape=[y_dim], name="p_{6}")

    x = torch.from_numpy(np.random.random((batch_n, x_dim)).astype("float32"))
    y = torch.from_numpy(np.random.random((batch_n, y_dim)).astype("float32"))
    a = torch.from_numpy(np.random.random((batch_n, a_dim)).astype("float32"))

    # In[4]:

    p1 = P1()
    p2 = P2()
    p3 = p2 * p1
    p3.name = "p_{3}"
    p5 = p3 * p4
    p5.name = "p_{5}"
    p_all = p1 * p2 * p4 * p6
    p_all.name = "p_{all}"

    # In[5]:

    print(p1)
    print_latex(p1)

    # In[6]:

    print(p2)
    print_latex(p2)

    # In[7]:

    print(p3)
    print_latex(p3)

    # In[8]:

    print(p4)
    print_latex(p4)

    # In[9]:

    print(p5)
    print_latex(p5)

    # In[10]:

    print(p_all)
    print_latex(p_all)

    # In[11]:

    for param in p3.parameters():
        print(type(param.data), param.size())

    # In[12]:

    p1.sample({"a": a, "y": y}, return_all=False)

    # In[13]:

    p1.sample({"a": a, "y": y}, sample_shape=[5], return_all=False)

    # In[14]:

    p1.sample({"a": a, "y": y}, return_all=True)

    # In[15]:

    p1_log_prob = p1.log_prob()
    print(p1_log_prob)
    print_latex(p1_log_prob)

    # In[16]:

    outputs = p1.sample({"y": y, "a": a})
    print(p1_log_prob.eval(outputs))

    # In[17]:

    outputs = p2.sample({"x": x, "y": y})
    print(p2.log_prob().eval(outputs))

    # In[18]:

    outputs = p1.sample({"y": y, "a": a})
    print(outputs)

    # In[19]:

    p2.sample(outputs)

    # In[20]:

    outputs = p3.sample({"y": y, "a": a}, batch_n=batch_n)
    print(p3.log_prob().eval(outputs))

    # In[21]:

    outputs = p_all.sample(batch_n=batch_n)
    print(p_all.log_prob().eval(outputs))

    # In[ ]:


# # Generative adversarial network (using the GAN class)
def test_run_gan():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:
    # In[4]:

    x_dim = 784
    z_dim = 100

    # generator model p(x|z)
    class Generator(Deterministic):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(z_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, x_dim),
                nn.Sigmoid()
            )

        def forward(self, z):
            x = self.model(z)
            return {"x": x}

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # generative model
    p_g = Generator()
    p = (p_g * prior).marginalize_var("z").to(device)

    # In[5]:

    print(p)
    print_latex(p)

    # In[6]:

    # discriminator model p(t|x)
    class Discriminator(Deterministic):
        def __init__(self):
            super(Discriminator, self).__init__(cond_var=["x"], var=["t"], name="d")

            self.model = nn.Sequential(
                nn.Linear(x_dim, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            t = self.model(x)
            return {"t": t}

    d = Discriminator().to(device)

    # In[7]:

    print(d)
    print_latex(d)

    # In[8]:

    model = GAN(p, d,
                optimizer=optim.Adam, optimizer_params={"lr": 0.0002},
                d_optimizer=optim.Adam, d_optimizer_params={"lr": 0.0002})
    print(model)
    print_latex(model)

    # In[9]:

    def train(epoch):
        train_loss = 0
        train_d_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss, d_loss = model.train({"x": x})
            train_loss += loss
            train_d_loss += d_loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        train_d_loss = train_d_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}, {:.4f}'.format(epoch, train_loss.item(), train_d_loss.item()))
        return train_loss

    # In[10]:

    def test(epoch):
        test_loss = 0
        test_d_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss, d_loss = model.test({"x": x})
            test_loss += loss
            test_d_loss += d_loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        test_d_loss = test_d_loss * test_loader.batch_size / len(test_loader.dataset)

        print('Test loss: {:.4f}, {:.4f}'.format(test_loss, test_d_loss.item()))
        return test_loss

    # In[11]:

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p_g.sample({"z": z_sample})["x"].view(-1, 1, 28, 28).cpu()
            return sample

    # In[12]:

    # writer = SummaryWriter()

    z_sample = torch.randn(64, z_dim).to(device)
    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = _y.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    # 
    # writer.close()

    # In[ ]:


# # Glow （CIFAR10）
def test_run_glow():
    # In[1]:
    # In[2]:

    root = '../data'
    num_workers = 8

    # transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])
    # 
    # train_loader = DataLoader(datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train),
    #                           batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 
    # test_loader = DataLoader(datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test),
    #                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(mock_cifar10, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(mock_cifar10, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # In[3]:
    # In[4]:

    in_channels = 3
    mid_channels = 64
    num_scales = 2
    input_dim = 32

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[in_channels, input_dim, input_dim], name="p_prior")

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_channels, mid_channels):
            super().__init__()
            self.resnet = ResNet(in_channels=in_channels, mid_channels=mid_channels, out_channels=in_channels * 2,
                                 num_blocks=8, kernel_size=3, padding=1,
                                 double_after_norm=True)

        def forward(self, x):
            s_t = self.resnet(x)
            log_s, t = torch.chunk(s_t, 2, dim=1)
            log_s = torch.tanh(log_s)
            return log_s, t

    # In[7]:

    flow_list = []

    flow_list.append(Preprocess())

    # Squeeze -> 3x coupling (channel-wise)
    flow_list.append(Squeeze())

    for i in range(3):
        flow_list.append(ActNorm2d(in_channels * 4))
        flow_list.append(ChannelConv(in_channels * 4))
        flow_list.append(AffineCoupling(in_features=in_channels * 4, mask_type="channel_wise",
                                        scale_translate_net=ScaleTranslateNet(in_channels * 4, mid_channels * 2),
                                        inverse_mask=False))
    flow_list.append(Unsqueeze())

    f = FlowList(flow_list)

    # In[9]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"]).to(device)
    print(p)
    print_latex(p)

    # In[10]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0

        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.inverse(z_sample).cpu()
            return sample

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = p.forward(x, compute_jacobian=False)
            recon_batch = p.inverse(z)

            comparison = torch.cat([x.view(-1, 3, 32, 32), recon_batch]).cpu()
            return comparison

    # In[13]:

    # writer = SummaryWriter()

    z_sample = torch.randn(64, 3, 32, 32).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:


# # Gaussian Mixture Model
def test_run_gmm():
    # In[1]:
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from mpl_toolkits.mplot3d import Axes3D

    # ### toy dataset

    # In[2]:

    # https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
    def sample(mu, var, nb_samples=500):
        """
        Return a tensor of (nb_samples, features), sampled
        from the parameterized gaussian.
        :param mu: torch.Tensor of the means
        :param var: torch.Tensor of variances (NOTE: zero covars.)
        """
        out = []
        for i in range(nb_samples):
            out += [
                torch.normal(mu, var.sqrt())
            ]
        return torch.stack(out, dim=0)

    # generate some clusters
    cluster1 = sample(
        torch.Tensor([1.5, 2.5]),
        torch.Tensor([1.2, .8]),
        nb_samples=150
    )

    cluster2 = sample(
        torch.Tensor([7.5, 7.5]),
        torch.Tensor([.75, .5]),
        nb_samples=50
    )

    cluster3 = sample(
        torch.Tensor([8, 1.5]),
        torch.Tensor([.6, .8]),
        nb_samples=100
    )

    def plot_2d_sample(sample_dict):
        x = sample_dict["x"][:, 0].data.numpy()
        y = sample_dict["x"][:, 1].data.numpy()
        # plt.plot(x, y, 'gx')

        # plt.show()

    # In[3]:

    # create the dummy dataset, by combining the clusters.
    samples = torch.cat([cluster1, cluster2, cluster3])
    samples = (samples - samples.mean(dim=0)) / samples.std(dim=0)
    samples_dict = {"x": samples}

    plot_2d_sample(samples_dict)

    # ## GMM

    # In[4]:

    z_dim = 3  # the number of mixture
    x_dim = 2

    distributions = []
    for i in range(z_dim):
        loc = torch.randn(x_dim)
        scale = torch.empty(x_dim).fill_(0.6)
        distributions.append(Normal(loc=loc, scale=scale, var=["x"], name="p_%d" % i))

    probs = torch.empty(z_dim).fill_(1. / z_dim)
    prior = Categorical(probs=probs, var=["z"], name="p_{prior}")

    # In[5]:

    p = MixtureModel(distributions=distributions, prior=prior)
    print(p)
    print_latex(p)

    # In[6]:

    post = p.posterior()
    print(post)
    print_latex(post)

    # In[7]:

    def get_density(N=200, x_range=(-5, 5), y_range=(-5, 5)):
        x = np.linspace(*x_range, N)
        y = np.linspace(*y_range, N)
        x, y = np.meshgrid(x, y)

        # get the design matrix
        points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        points = torch.from_numpy(points).float()

        pdf = p.prob().eval({"x": points}).data.numpy().reshape([N, N])

        return x, y, pdf

    # In[8]:

    # def plot_density_3d(x, y, loglike):
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.gca(projection='3d')
    #     ax.plot_surface(x, y, loglike, rstride=3, cstride=3, linewidth=1, antialiased=True,
    #                     cmap=cm.inferno)
    #     cset = ax.contourf(x, y, loglike, zdir='z', offset=-0.15, cmap=cm.inferno)
    # 
    #     # adjust the limits, ticks and view angle
    #     ax.set_zlim(-0.15, 0.2)
    #     ax.set_zticks(np.linspace(0, 0.2, 5))
    #     ax.view_init(27, -21)
    #     plt.show()

    # In[9]:

    def plot_density_2d(x, y, pdf):
        # fig = plt.figure(figsize=(5, 5))

        # plt.plot(samples_dict["x"][:, 0].data.numpy(), samples_dict["x"][:, 1].data.numpy(), 'gx')
        # 
        # for d in distributions:
        #     plt.scatter(d.loc[0, 0], d.loc[0, 1], c='r', marker='o')
        # 
        # cs = plt.contour(x, y, pdf, 10, colors='k', linewidths=2)
        # plt.show()
        pass

    # In[10]:

    eps = 1e-6
    min_scale = 1e-6

    # plot_density_3d(*get_density())
    plot_density_2d(*get_density())
    print("Epoch: {}, log-likelihood: {}".format(0, p.log_prob().mean().eval(samples_dict)))
    for epoch in range(20):
        # E-step
        posterior = post.prob().eval(samples_dict)

        # M-step
        N_k = posterior.sum(dim=1)  # (n_mix,)

        # update probs
        probs = N_k / N_k.sum()  # (n_mix,)
        prior.probs[0] = probs

        # update loc & scale
        loc = (posterior[:, None] @ samples[None]).squeeze(1)  # (n_mix, n_dim)
        loc /= (N_k[:, None] + eps)

        cov = (samples[None, :, :] - loc[:, None, :]) ** 2  # Covariances are set to 0.
        var = (posterior[:, None, :] @ cov).squeeze(1)  # (n_mix, n_dim)
        var /= (N_k[:, None] + eps)
        scale = var.sqrt()

        for i, d in enumerate(distributions):
            d.loc[0] = loc[i]
            d.scale[0] = scale[i]

        #    plot_density_3d(*get_density())
        plot_density_2d(*get_density())
        print("Epoch: {}, log-likelihood: {}".format(epoch + 1, p.log_prob().mean().eval({"x": samples}).mean()))

    # In[11]:

    psudo_sample_dict = p.sample(batch_n=200)
    plot_2d_sample(samples_dict)

    # In[ ]:


# # Variational inference on a hierarchical latent model
def test_run_hvi():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    # In[4]:

    x_dim = 784
    a_dim = 64
    z_dim = 32

    # inference models
    class Q1(Normal):
        def __init__(self):
            super(Q1, self).__init__(cond_var=["x"], var=["a"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, a_dim)
            self.fc32 = nn.Linear(512, a_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    class Q2(Normal):
        def __init__(self):
            super(Q2, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    q1 = Q1().to(device)
    q2 = Q2().to(device)

    q = q1 * q2
    q.name = "q"

    # generative models
    class P2(Normal):
        def __init__(self):
            super(P2, self).__init__(cond_var=["z"], var=["a"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, a_dim)
            self.fc32 = nn.Linear(512, a_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    class P3(Bernoulli):
        def __init__(self):
            super(P3, self).__init__(cond_var=["a"], var=["x"], name="p")

            self.fc1 = nn.Linear(a_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, a):
            h = F.relu(self.fc1(a))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p2 = P2().to(device)
    p3 = P3().to(device)

    p1 = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    _p = p2 * p3
    p = _p * p1

    # In[5]:

    print(p)
    print_latex(p)

    # In[6]:

    print(_p)
    print_latex(_p)

    # In[7]:

    print(q)
    print_latex(q)

    # In[8]:

    model = VI(p, q, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[9]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[10]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[11]:

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = q.sample({"x": x})
            z = get_dict_values(z, _p.cond_var, return_dict=True)  # select latent variables
            recon_batch = _p.sample(z)["x"].view(-1, 1, 28, 28)  # TODO: it should be sample_mean

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = _p.sample({"z": z_sample})["x"].view(-1, 1, 28, 28).cpu()  # TODO: it should be sample_mean
            return sample

    # In[12]:

    # writer = SummaryWriter()

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()


# # JMVAE with a PoE encoder (using the VAE class)
# * JMVAE: Joint Multimodal Learning with Deep Generative Models
# * The PoE encoder is originally proposed in "Multimodal Generative Models for Scalable Weakly-Supervised Learning"
def test_run_jmvae_poe():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:
    # In[4]:

    x_dim = 784
    y_dim = 10
    z_dim = 64

    # inference model q(z|x)
    class InferenceX(Normal):
        def __init__(self):
            super(InferenceX, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # inference model q(z|y)
    class InferenceY(Normal):
        def __init__(self):
            super(InferenceY, self).__init__(cond_var=["y"], var=["z"], name="q")

            self.fc1 = nn.Linear(y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, y):
            h = F.relu(self.fc1(y))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class GeneratorX(Bernoulli):
        def __init__(self):
            super(GeneratorX, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    # generative model p(y|z)    
    class GeneratorY(Categorical):
        def __init__(self):
            super(GeneratorY, self).__init__(cond_var=["z"], var=["y"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, y_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": F.softmax(self.fc3(h), dim=1)}

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    p_x = GeneratorX().to(device)
    p_y = GeneratorY().to(device)
    p = p_x * p_y

    q_x = InferenceX().to(device)
    q_y = InferenceY().to(device)

    q = ProductOfNormal([q_x, q_y], name="q").to(device)

    # In[5]:

    print(q)
    print_latex(q)

    # In[6]:

    print(p)
    print_latex(p)

    # In[7]:

    kl = KullbackLeibler(q, prior)
    kl_x = KullbackLeibler(q, q_x)
    kl_y = KullbackLeibler(q, q_y)

    regularizer = kl + kl_x + kl_y
    print(regularizer)
    print_latex(regularizer)

    # In[8]:

    model = VAE(q, p, other_distributions=[q_x, q_y],
                regularizer=regularizer, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[9]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[10]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[11]:

    def plot_reconstrunction_missing(x):
        with torch.no_grad():
            z = q_x.sample({"x": x}, return_all=False)
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_label(x, y):
        with torch.no_grad():
            x_all = [x.view(-1, 1, 28, 28)]
            for i in range(7):
                z = q_y.sample({"y": y}, return_all=False)
                recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)
                x_all.append(recon_batch)

            comparison = torch.cat(x_all).cpu()
            return comparison

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            z = q.sample({"x": x, "y": y}, return_all=False)
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    # In[12]:

    # writer = SummaryWriter()

    plot_number = 1

    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_label(_x[:8], _y[:8])
        recon_missing = plot_reconstrunction_missing(_x[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_label', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_missing', recon_missing, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Joint multimodal variational autoencoder (JMVAE, using the VAE class)
# Original paper: Joint Multimodal Learning with Deep Generative Models (https://arxiv.org/abs/1611.01891 )
def test_run_jmvae():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:
    # In[4]:

    x_dim = 784
    y_dim = 10
    z_dim = 64

    # inference model q(z|x,y)
    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x", "y"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim + y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x, y):
            h = F.relu(self.fc1(torch.cat([x, y], 1)))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # inference model q(z|x)
    class InferenceX(Normal):
        def __init__(self):
            super(InferenceX, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # inference model q(z|y)
    class InferenceY(Normal):
        def __init__(self):
            super(InferenceY, self).__init__(cond_var=["y"], var=["z"], name="q")

            self.fc1 = nn.Linear(y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, y):
            h = F.relu(self.fc1(y))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class GeneratorX(Bernoulli):
        def __init__(self):
            super(GeneratorX, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    # generative model p(y|z)    
    class GeneratorY(Categorical):
        def __init__(self):
            super(GeneratorY, self).__init__(cond_var=["z"], var=["y"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, y_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": F.softmax(self.fc3(h), dim=1)}

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    p_x = GeneratorX().to(device)
    p_y = GeneratorY().to(device)

    q = Inference().to(device)
    q_x = InferenceX().to(device)
    q_y = InferenceY().to(device)

    p = p_x * p_y

    # In[5]:

    print(p)
    print_latex(p)

    # In[6]:

    kl = KullbackLeibler(q, prior)
    kl_x = KullbackLeibler(q, q_x)
    kl_y = KullbackLeibler(q, q_y)

    regularizer = kl + kl_x + kl_y
    print(regularizer)
    print_latex(regularizer)

    # In[7]:

    model = VAE(q, p, other_distributions=[q_x, q_y],
                regularizer=regularizer, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[8]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[9]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[10]:

    def plot_reconstrunction_missing(x):
        with torch.no_grad():
            z = q_x.sample({"x": x}, return_all=False)
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_label(x, y):
        with torch.no_grad():
            x_all = [x.view(-1, 1, 28, 28)]
            for i in range(7):
                z = q_y.sample({"y": y}, return_all=False)
                recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)
                x_all.append(recon_batch)

            comparison = torch.cat(x_all).cpu()
            return comparison

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            z = q.sample({"x": x, "y": y}, return_all=False)
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    # In[11]:

    # writer = SummaryWriter()

    plot_number = 1

    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_label(_x[:8], _y[:8])
        recon_missing = plot_reconstrunction_missing(_x[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_label', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_missing', recon_missing, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Semi-supervised learning with M2 model
def test_run_m2():
    # In[1]:
    # In[2]:

    # https://github.com/wohlert/semi-supervised-pytorch/blob/master/examples/notebooks/datautils.py

    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    # from torchvision.datasets import MNIST
    import numpy as np

    labels_per_class = 10
    n_labels = 10

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    #
    # mnist_train = MNIST(root=root, train=True, download=True, transform=transform)
    # mnist_valid = MNIST(root=root, train=False, transform=transform)
    mnist_train = mock_mnist
    mnist_valid = mock_mnist

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    # labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
    #                                        sampler=get_sampler(mnist_train.targets.numpy(), labels_per_class),
    #                                        **kwargs)
    # unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
    #                                          sampler=get_sampler(mnist_train.targets.numpy()), **kwargs)
    # validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size,
    #                                          sampler=get_sampler(mnist_valid.targets.numpy()), **kwargs)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                           sampler=get_sampler(mock_mnist_targets.numpy(), labels_per_class),
                                           **kwargs)
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                             sampler=get_sampler(mock_mnist_targets.numpy()), **kwargs)
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size,
                                             sampler=get_sampler(mock_mnist_targets.numpy()), **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, Bernoulli, RelaxedCategorical, Categorical
    from pixyz.models import Model
    from pixyz.losses import ELBO
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 784
    y_dim = 10
    z_dim = 64

    # inference model q(z|x,y)
    class Inference(Normal):
        def __init__(self):
            super().__init__(cond_var=["x", "y"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim + y_dim, 512)
            self.fc21 = nn.Linear(512, z_dim)
            self.fc22 = nn.Linear(512, z_dim)

        def forward(self, x, y):
            h = F.relu(self.fc1(torch.cat([x, y], 1)))
            return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

    # generative model p(x|z,y)
    class Generator(Bernoulli):
        def __init__(self):
            super().__init__(cond_var=["z", "y"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim + y_dim, 512)
            self.fc2 = nn.Linear(512, x_dim)

        def forward(self, z, y):
            h = F.relu(self.fc1(torch.cat([z, y], 1)))
            return {"probs": torch.sigmoid(self.fc2(h))}

    # classifier p(y|x)
    class Classifier(RelaxedCategorical):
        def __init__(self):
            super(Classifier, self).__init__(cond_var=["x"], var=["y"], name="p")
            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, y_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.softmax(self.fc2(h), dim=1)
            return {"probs": h}

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # distributions for supervised learning
    p = Generator().to(device)
    q = Inference().to(device)
    f = Classifier().to(device)
    p_joint = p * prior

    # In[5]:

    print(p_joint)
    print_latex(p_joint)

    # In[6]:

    print(q)
    print_latex(q)

    # In[7]:

    print(f)
    print_latex(f)

    # In[8]:

    # distributions for unsupervised learning
    _q_u = q.replace_var(x="x_u", y="y_u")
    p_u = p.replace_var(x="x_u", y="y_u")
    f_u = f.replace_var(x="x_u", y="y_u")

    q_u = _q_u * f_u
    p_joint_u = p_u * prior

    p_joint_u.to(device)
    q_u.to(device)
    f_u.to(device)

    print(p_joint_u)
    print_latex(p_joint_u)

    # In[9]:

    print(q_u)
    print_latex(q_u)

    # In[10]:

    print(f_u)
    print_latex(f_u)

    # In[11]:

    elbo_u = ELBO(p_joint_u, q_u)
    elbo = ELBO(p_joint, q)
    nll = -f.log_prob()  # or -LogProb(f)

    rate = 1 * (len(unlabelled) + len(labelled)) / len(labelled)

    loss_cls = -elbo_u.mean() - elbo.mean() + (rate * nll).mean()
    print(loss_cls)
    print_latex(loss_cls)

    # In[12]:

    model = Model(loss_cls, test_loss=nll.mean(),
                  distributions=[p, q, f], optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[13]:

    def train(epoch):
        train_loss = 0
        for x_u, y_u in tqdm(unlabelled):
            x, y = iter(labelled).next()
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            x_u = x_u.to(device)
            loss = model.train({"x": x, "y": y, "x_u": x_u})
            train_loss += loss

        train_loss = train_loss * unlabelled.batch_size / len(unlabelled.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))

        return train_loss

    # In[14]:

    def test(epoch):
        test_loss = 0
        correct = 0
        total = 0
        for x, y in validation:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

            pred_y = f.sample_mean({"x": x})
            total += y.size(0)
            correct += (pred_y.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        test_loss = test_loss * validation.batch_size / len(validation.dataset)
        test_accuracy = 100 * correct / total
        print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_accuracy))
        return test_loss, test_accuracy

    # In[15]:

    # writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss, test_accuracy = test(epoch)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    #     writer.add_scalar('test_accuracy', test_accuracy, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Maximum likelihood estimation (using the ML class)
def test_run_maximum_likelihood():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Categorical
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 784
    y_dim = 10

    # classifier p(y|x)
    class Classifier(Categorical):
        def __init__(self):
            super(Classifier, self).__init__(cond_var=["x"], var=["y"])
            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, y_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.softmax(self.fc3(h), dim=1)

            return {"probs": h}

    p = Classifier().to(device)

    # In[5]:

    print(p)
    print_latex(p)

    # In[6]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[7]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[8]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[9]:

    # writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # MMD-VAE (using the Model class)
def test_run_mmd_vae():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, Bernoulli, DataDistribution
    from pixyz.losses import CrossEntropy, MMD
    from pixyz.models import Model
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 784
    z_dim = 64

    # inference model q(z|x)
    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class Generator(Bernoulli):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    p_data = DataDistribution(["x"]).to(device)
    q_mg = (q * p_data).marginalize_var("x")
    q_mg.name = "q"

    # In[5]:

    print(p)
    print_latex(p)

    # In[6]:

    print(q_mg)
    print_latex(q_mg)

    # In[7]:

    loss_cls = CrossEntropy(q, p).mean() + MMD(q_mg, prior, kernel="gaussian", sigma_sqr=z_dim / 2.)
    print(loss_cls)
    print_latex(loss_cls)

    # In[8]:

    model = Model(loss=loss_cls, distributions=[p, q, q_mg], optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[9]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[10]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[11]:

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = q.sample({"x": x}, return_all=False)
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.sample_mean({"z": z_sample}).view(-1, 1, 28, 28).cpu()
            return sample

    # In[12]:

    # writer = SummaryWriter()

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # MVAE
def test_run_mvae():
    # * Original paper: Multimodal Generative Models for Scalable Weakly-Supervised Learning (https://papers.nips.cc/paper/7801-multimodal-generative-models-for-scalable-weakly-supervised-learning.pdf)
    # * Original code: https://github.com/mhw32/multimodal-vae-public
    # 

    # ### MVAE summary  
    # Multimodal variational autoencoder(MVAE) uses a product-of-experts inferece network and a sub-sampled training paradigm to solve the multi-modal inferece problem.  
    # - Product-of-experts  
    # In the multimodal setting we assume the N modalities, $x_{1}, x_{2}, ..., x_{N}$, are conditionally independent given the common latent variable, z. That is we assume a generative model of the form $p_{\theta}(x_{1}, x_{2}, ..., x_{N}, z) = p(z)p_{\theta}(x_{1}|z)p_{\theta}(x_{2}|z)$・・・$p_{\theta}(x_{N}|z)$. The conditional independence assumptions in the generative model imply a relation among joint- and simgle-modality posteriors. That is, the joint posterior is a procuct of individual posteriors, with an additional quotient by the prior.  
    # 
    # - Sub-sampled training  
    # MVAE sub-sample which ELBO terms to optimize for every gradient step for capturing the relationships between modalities and training individual inference networks.  

    # In[1]:
    # In[2]:

    # MNIST
    # treat labels as a second modality
    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.utils import print_latex

    # ## Define probability distributions
    # ### In the original paper
    # Modalities: $x_{1}, x_{2}, ..., x_{N}$  
    # Generative model:  
    # 
    # $p_{\theta}\left(x_{1}, x_{2}, \ldots, x_{N}, z\right)=p(z) p_{\theta}\left(x_{1} | z\right) p_{\theta}\left(x_{2} | z\right) \cdots p_{\theta}\left(x_{N} | z\right)$  
    # 
    # Inference:  
    # 
    # $p\left(z | x_{1}, \ldots, x_{N}\right) \propto \frac{\prod_{i=1}^{N} p\left(z | x_{i}\right)}{\prod_{i=1}^{N-1} p(z)} \approx \frac{\prod_{i=1}^{N}\left[\tilde{q}\left(z | x_{i}\right) p(z)\right]}{\prod_{i=1}^{N-1} p(z)}=p(z) \prod_{i=1}^{N} \tilde{q}\left(z | x_{i}\right)$  
    # 
    # ### MNIST settings
    # Modalities:
    # - x for image modality
    # - y for label modality
    # 
    # Prior: $p(z) = \cal N(z; \mu=0, \sigma^2=1)$  
    # Generators:  
    # $p_{\theta}(x|z) = \cal B(x; \lambda = g_x(z))$ for image modality  
    # $p_{\theta}(y|z) = \cal Cat(y; \lambda = g_y(z))$ for label modality  
    # $p_{\theta}\left(x, y, z\right)=p(z) p_{\theta}(x| z) p_{\theta}(y | z)$
    # 
    # Inferences:  
    # $q_{\phi}(z|x) = \cal N(z; \mu=fx_\mu(x), \sigma^2=fx_{\sigma^2}(x))$ for image modality  
    # $q_{\phi}(z|y) = \cal N(z; \mu=fy_\mu(y), \sigma^2=fy_{\sigma^2}(y))$ for label modality  
    # $p(z)q_{\phi}(z|x)q_{\phi}(z|y)$
    # 

    # In[4]:

    from pixyz.distributions import Normal, Bernoulli, Categorical, ProductOfNormal

    x_dim = 784
    y_dim = 10
    z_dim = 64

    # inference model q(z|x) for image modality
    class InferenceX(Normal):
        def __init__(self):
            super(InferenceX, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # inference model q(z|y) for label modality
    class InferenceY(Normal):
        def __init__(self):
            super(InferenceY, self).__init__(cond_var=["y"], var=["z"], name="q")

            self.fc1 = nn.Linear(y_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, y):
            h = F.relu(self.fc1(y))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z) 
    class GeneratorX(Bernoulli):
        def __init__(self):
            super(GeneratorX, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    # generative model p(y|z)    
    class GeneratorY(Categorical):
        def __init__(self):
            super(GeneratorY, self).__init__(cond_var=["z"], var=["y"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, y_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": F.softmax(self.fc3(h), dim=1)}

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    p_x = GeneratorX().to(device)
    p_y = GeneratorY().to(device)
    p = p_x * p_y

    q_x = InferenceX().to(device)
    q_y = InferenceY().to(device)

    # equation (4) in the paper
    # "we can use a product of experts (PoE), including a “prior expert”, as the approximating distribution for the joint-posterior"
    # Pixyz docs: https://docs.pixyz.io/en/latest/distributions.html#pixyz.distributions.ProductOfNormal
    q = ProductOfNormal([q_x, q_y], name="q").to(device)

    # In[5]:

    print(q)
    print_latex(q)

    # In[6]:

    print(p)
    print_latex(p)

    # ## Define Loss function
    # $\cal L = \mathrm{ELBO}\left(x_{1}, \ldots, x_{N}\right)+\sum_{i=1}^{N} \mathrm{ELBO}\left(x_{i}\right)+\sum_{j=1}^{k} \mathrm{ELBO}\left(X_{j}\right)$

    # In[7]:

    from pixyz.losses import KullbackLeibler
    from pixyz.losses import LogProb
    from pixyz.losses import Expectation as E

    # In[8]:

    ELBO = -E(q, LogProb(p)) + KullbackLeibler(q, prior)
    ELBO_x = -E(q_x, LogProb(p_x)) + KullbackLeibler(q_x, prior)
    ELBO_y = -E(q_y, LogProb(p_y)) + KullbackLeibler(q_y, prior)

    loss = ELBO.mean() + ELBO_x.mean() + ELBO_y.mean()
    print_latex(loss)  # Note: Terms in the printed loss may be reordered

    # ## Define MVAE model using Model Class

    # In[9]:

    from pixyz.models import Model

    model = Model(loss=loss, distributions=[p_x, p_y, q_x, q_y],
                  optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # ## Define Train and Test loop using model

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # ## Reconstruction and generation

    # In[12]:

    def plot_reconstrunction_missing_label_modality(x):
        with torch.no_grad():
            # infer from x (image modality) only
            z = q_x.sample({"x": x}, return_all=False)
            # generate image from latent variable
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_label(x, y):
        with torch.no_grad():
            x_all = [x.view(-1, 1, 28, 28)]
            for i in range(7):
                # infer from y (label modality) only
                z = q_y.sample({"y": y}, return_all=False)

                # generate image from latent variable
                recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)
                x_all.append(recon_batch)

            comparison = torch.cat(x_all).cpu()
            return comparison

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            # infer from x and y
            z = q.sample({"x": x, "y": y}, return_all=False)
            # generate image from latent variable
            recon_batch = p_x.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    # In[13]:

    # for visualising in TensorBoard
    # writer = SummaryWriter()

    plot_number = 1

    # set-aside observation for watching generative model improvement 
    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_label(_x[:8], _y[:8])
        recon_missing = plot_reconstrunction_missing_label_modality(_x[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_label', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_missing_label', recon_missing, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # A toy example of variational inference with normalizing flow (using the VI class)
def test_run_normalizing_flow_toy():
    # In[1]:
    # In[2]:

    from pixyz.distributions import CustomProb, Normal, TransformedDistribution
    from pixyz.models import VI
    from pixyz.flows import PlanarFlow, FlowList
    from pixyz.utils import print_latex

    # In[3]:

    # def plot_samples(points):
    #     X_LIMS = (-4, 4)
    #     Y_LIMS = (-4, 4)
    # 
    #     fig = plt.figure(figsize=(4, 4))
    #     ax = fig.add_subplot(111)
    #     ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25)
    #     ax.set_xlim(*X_LIMS)
    #     ax.set_ylim(*Y_LIMS)
    #     ax.set_xlabel("p(z)")
    # 
    #     plt.show()

    # In[4]:

    import torch

    x_dim = 2

    def log_prob(z):
        z1, z2 = torch.chunk(z, chunks=2, dim=1)
        norm = torch.sqrt(z1 ** 2 + z2 ** 2)

        exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
        exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
        u = 0.5 * ((norm - 2) / 0.4) ** 2 - torch.log(exp1 + exp2)

        return -u

    p = CustomProb(log_prob, var=["z"])

    # In[5]:

    # def plot_density(p):
    #     X_LIMS = (-4, 4)
    #     Y_LIMS = (-4, 4)
    # 
    #     x1 = np.linspace(*X_LIMS, 300)
    #     x2 = np.linspace(*Y_LIMS, 300)
    #     x1, x2 = np.meshgrid(x1, x2)
    #     shape = x1.shape
    #     x1 = x1.ravel()
    #     x2 = x2.ravel()
    # 
    #     z = np.c_[x1, x2]
    #     z = torch.FloatTensor(z)
    # 
    #     density_values = p.prob().eval({"z": z}).data.numpy().reshape(shape)
    #     plt.imshow(density_values, cmap='jet')
    #     plt.show()

    # plot_density(p)

    # In[6]:

    # prior
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["x"], features_shape=[x_dim], name="prior").to(device)

    # In[7]:

    # flow
    f = FlowList([PlanarFlow(x_dim) for _ in range(32)])

    # In[8]:

    # transformed distribution (x -> f -> z)
    q = TransformedDistribution(prior, f, var=["z"], name="q").to(device)
    print(q)
    print_latex(q)

    # In[9]:

    model = VI(p, q, optimizer=optim.Adam, optimizer_params={"lr": 1e-2})
    print(model)
    print_latex(model)

    # In[10]:

    for epoch in range(epochs):
        loss = model.train(batch_size=batch_size)

        if epoch % 100 == 0:
            print('Epoch: {} Test loss: {:.4f}'.format(epoch, loss))

            loss = model.test(batch_n=batch_size)
            samples = q.sample(batch_n=1000)
            # plot_samples(samples["z"].cpu().data.numpy())

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Real NVP （CIFAR10）
def test_run_real_nvp_cifar():
    # In[1]:
    # In[2]:

    # root = '../data'
    # num_workers = 8
    # 
    # transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])
    # 
    # train_loader = DataLoader(datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train),
    #                           batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 
    # test_loader = DataLoader(datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test),
    #                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_cifar10, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_cifar10, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, InverseTransformedDistribution
    from pixyz.flows import AffineCoupling, FlowList, Squeeze, Unsqueeze, Preprocess, Flow
    from pixyz.layers import ResNet
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    in_channels = 3
    mid_channels = 64
    num_scales = 2
    input_dim = 32

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[in_channels, input_dim, input_dim], name="p_prior")

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_channels, mid_channels):
            super().__init__()
            self.resnet = ResNet(in_channels=in_channels, mid_channels=mid_channels, out_channels=in_channels * 2,
                                 num_blocks=8, kernel_size=3, padding=1,
                                 double_after_norm=True)

        def forward(self, x):
            s_t = self.resnet(x)
            log_s, t = torch.chunk(s_t, 2, dim=1)
            log_s = torch.tanh(log_s)
            return log_s, t

    # In[7]:

    flow_list = [Preprocess()]

    # Coupling_Layer(checkboard) x3
    for i in range(3):
        flow_list.append(AffineCoupling(in_features=in_channels, mask_type="checkerboard",
                                        scale_translate_net=ScaleTranslateNet(in_channels, mid_channels),
                                        inverse_mask=(i % 2 != 0)))

    # Squeeze -> 3x coupling (channel-wise)
    flow_list.append(Squeeze())

    for i in range(3):
        flow_list.append(AffineCoupling(in_features=in_channels * 4, mask_type="channel_wise",
                                        scale_translate_net=ScaleTranslateNet(in_channels * 4, mid_channels * 2),
                                        inverse_mask=(i % 2 != 0)))
    flow_list.append(Unsqueeze())

    f = FlowList(flow_list)

    # In[8]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"]).to(device)
    print_latex(p)

    # In[9]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0

        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.inverse(z_sample).cpu()
            return sample

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = p.forward(x, compute_jacobian=False)
            recon_batch = p.inverse(z)

            comparison = torch.cat([x.view(-1, 3, 32, 32), recon_batch]).cpu()
            return comparison

    # In[13]:

    # writer = SummaryWriter()

    z_sample = torch.randn(64, 3, 32, 32).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Real NVP （CIFAR10）
def test_run_real_nvp_cond():
    # In[1]:
    # In[2]:

    # root = '../data'
    # num_workers = 8
    # 
    # transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])
    # 
    # train_loader = DataLoader(datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train),
    #                           batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 
    # test_loader = DataLoader(datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test),
    #                          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_cifar10, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_cifar10, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, InverseTransformedDistribution
    from pixyz.flows import AffineCoupling, FlowList, Squeeze, Unsqueeze, Preprocess, Flow
    from pixyz.layers import ResNet
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    in_channels = 3
    mid_channels = 64
    num_scales = 2
    input_dim = 32

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[in_channels, input_dim, input_dim], name="p_prior")

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_channels, mid_channels):
            super().__init__()
            self.resnet = ResNet(in_channels=in_channels, mid_channels=mid_channels, out_channels=in_channels * 2,
                                 num_blocks=8, kernel_size=3, padding=1,
                                 double_after_norm=True)

        def forward(self, x):
            s_t = self.resnet(x)
            log_s, t = torch.chunk(s_t, 2, dim=1)
            log_s = torch.tanh(log_s)
            return log_s, t

    # In[7]:

    flow_list = [Preprocess()]

    # Coupling_Layer(checkboard) x3
    for i in range(3):
        flow_list.append(AffineCoupling(in_features=in_channels, mask_type="checkerboard",
                                        scale_translate_net=ScaleTranslateNet(in_channels, mid_channels),
                                        inverse_mask=(i % 2 != 0)))

    # Squeeze -> 3x coupling (channel-wise)
    flow_list.append(Squeeze())

    for i in range(3):
        flow_list.append(AffineCoupling(in_features=in_channels * 4, mask_type="channel_wise",
                                        scale_translate_net=ScaleTranslateNet(in_channels * 4, mid_channels * 2),
                                        inverse_mask=(i % 2 != 0)))
    flow_list.append(Unsqueeze())

    f = FlowList(flow_list)

    # In[8]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"]).to(device)
    print_latex(p)

    # In[9]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0

        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.inverse(z_sample).cpu()
            return sample

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = p.forward(x, compute_jacobian=False)
            recon_batch = p.inverse(z)

            comparison = torch.cat([x.view(-1, 3, 32, 32), recon_batch]).cpu()
            return comparison

    # In[13]:

    # writer = SummaryWriter()

    z_sample = torch.randn(64, 3, 32, 32).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Conditional Real NVP
def test_run_real_nvp_cond_():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, InverseTransformedDistribution
    from pixyz.flows import AffineCoupling, FlowList, BatchNorm1d, Shuffle, Preprocess, Reverse
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 28 * 28
    y_dim = 10
    z_dim = x_dim

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_prior").to(device)

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features + y_dim, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.fc3_s = nn.Linear(hidden_features, in_features)
            self.fc3_t = nn.Linear(hidden_features, in_features)

        def forward(self, x, y):
            hidden = F.relu(self.fc2(F.relu(self.fc1(torch.cat([x, y], 1)))))
            log_s = torch.tanh(self.fc3_s(hidden))
            t = self.fc3_t(hidden)
            return log_s, t

    # In[7]:

    # flow
    flow_list = []
    num_block = 5

    flow_list.append(Preprocess())

    for i in range(num_block):
        flow_list.append(AffineCoupling(in_features=x_dim,
                                        scale_translate_net=ScaleTranslateNet(x_dim, 1028),
                                        inverse_mask=(i % 2 != 0)))

        flow_list.append(BatchNorm1d(x_dim))

    f = FlowList(flow_list)

    # In[8]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"], cond_var=["y"]).to(device)
    print_latex(p)

    # In[9]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            z = p.forward(x, y, compute_jacobian=False)
            recon_batch = p.inverse(z, y).view(-1, 1, 28, 28)

            recon = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return recon

    def plot_image_from_latent(z, y):
        with torch.no_grad():
            sample = p.inverse(z, y).view(-1, 1, 28, 28).cpu()
            return sample

    def plot_reconstrunction_changing_y(x, y):
        y_change = torch.eye(10)[range(7)].to(device)
        batch_dummy = torch.ones(x.size(0))[:, None].to(device)
        recon_all = []

        with torch.no_grad():
            for _y in y_change:
                z = p.forward(x, y, compute_jacobian=False)
                recon_batch = p.inverse(z, batch_dummy * _y[None, :]).view(-1, 1, 28, 28)
                recon_all.append(recon_batch)

            recon_changing_y = torch.cat(recon_all)
            recon_changing_y = torch.cat([x.view(-1, 1, 28, 28), recon_changing_y]).cpu()
            return recon_changing_y

    # In[13]:

    # writer = SummaryWriter()

    plot_number = 5

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    y_sample = torch.eye(10)[[plot_number] * 64].to(device)

    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_latent(z_sample, y_sample)
        recon_changing_y = plot_reconstrunction_changing_y(_x[:8], _y[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_change_y', recon_changing_y, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Conditional Real NVP
def test_run_real_nvp_cond__():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, InverseTransformedDistribution
    from pixyz.flows import AffineCoupling, FlowList, BatchNorm1d, Shuffle, Preprocess, Reverse
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 28 * 28
    y_dim = 10
    z_dim = x_dim

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_prior").to(device)

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features + y_dim, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.fc3_s = nn.Linear(hidden_features, in_features)
            self.fc3_t = nn.Linear(hidden_features, in_features)

        def forward(self, x, y):
            hidden = F.relu(self.fc2(F.relu(self.fc1(torch.cat([x, y], 1)))))
            log_s = torch.tanh(self.fc3_s(hidden))
            t = self.fc3_t(hidden)
            return log_s, t

    # In[7]:

    # flow
    flow_list = []
    num_block = 5

    flow_list.append(Preprocess())

    for i in range(num_block):
        flow_list.append(AffineCoupling(in_features=x_dim,
                                        scale_translate_net=ScaleTranslateNet(x_dim, 1028),
                                        inverse_mask=(i % 2 != 0)))

        flow_list.append(BatchNorm1d(x_dim))

    f = FlowList(flow_list)

    # In[8]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"], cond_var=["y"]).to(device)
    print_latex(p)

    # In[9]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.train({"x": x, "y": y})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = torch.eye(10)[y].to(device)
            loss = model.test({"x": x, "y": y})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_reconstrunction(x, y):
        with torch.no_grad():
            z = p.forward(x, y, compute_jacobian=False)
            recon_batch = p.inverse(z, y).view(-1, 1, 28, 28)

            recon = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return recon

    def plot_image_from_latent(z, y):
        with torch.no_grad():
            sample = p.inverse(z, y).view(-1, 1, 28, 28).cpu()
            return sample

    def plot_reconstrunction_changing_y(x, y):
        y_change = torch.eye(10)[range(7)].to(device)
        batch_dummy = torch.ones(x.size(0))[:, None].to(device)
        recon_all = []

        with torch.no_grad():
            for _y in y_change:
                z = p.forward(x, y, compute_jacobian=False)
                recon_batch = p.inverse(z, batch_dummy * _y[None, :]).view(-1, 1, 28, 28)
                recon_all.append(recon_batch)

            recon_changing_y = torch.cat(recon_all)
            recon_changing_y = torch.cat([x.view(-1, 1, 28, 28), recon_changing_y]).cpu()
            return recon_changing_y

    # In[13]:

    # writer = SummaryWriter()

    plot_number = 5

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    y_sample = torch.eye(10)[[plot_number] * 64].to(device)

    _x, _y = iter(test_loader).next()
    _x = _x.to(device)
    _y = torch.eye(10)[_y].to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8], _y[:8])
        sample = plot_image_from_latent(z_sample, y_sample)
        recon_changing_y = plot_reconstrunction_changing_y(_x[:8], _y[:8])

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    #     writer.add_images('Image_reconstrunction_change_y', recon_changing_y, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # A toy example of Real NVP (using the ML class)
# def test_run_real_nvp_toy():
# 
#     # In[1]:
#     # In[2]:
# 
# 
#     from pixyz.distributions import Normal, InverseTransformedDistribution
#     from pixyz.flows import AffineCoupling, FlowList, BatchNorm1d
#     from pixyz.models import ML
#     from pixyz.utils import print_latex
# 
# 
#     # In[3]:
# 
# 
#     # def plot_samples(points, noise):
#     #     X_LIMS = (-1.5, 2.5)
#     #     Y_LIMS = (-2.5, 2.5)
#     # 
#     #     fig = plt.figure(figsize=(8, 4))
#     #     ax = fig.add_subplot(121)
#     #     ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25, c="b")
#     #     ax.set_xlim(*X_LIMS)
#     #     ax.set_ylim(*Y_LIMS)
#     #     ax.set_xlabel("p(x)")
#     # 
#     #     X_LIMS = (-3, 3)
#     #     Y_LIMS = (-3, 3)
#     # 
#     #     ax = fig.add_subplot(122)
#     #     ax.scatter(noise[:, 0], noise[:, 1], alpha=0.7, s=25, c="r")
#     #     ax.set_xlim(*X_LIMS)
#     #     ax.set_ylim(*Y_LIMS)
#     #     ax.set_xlabel("p(z)")
#     # 
#     #     plt.show()
# 
# 
#     # In[4]:
# 
# 
#     x_dim = 2
#     z_dim = x_dim
# 
#     # In[5]:
# 
# 
#     # prior
#     prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
#                    var=["z"], features_shape=[z_dim], name="prior").to(device)
# 
# 
#     # In[6]:
# 
# 
#     class ScaleTranslateNet(nn.Module):
#         def __init__(self, in_features, hidden_features):
#             super().__init__()
#             self.layers = nn.Sequential(nn.Linear(in_features, hidden_features),
#                                         nn.ReLU(),
#                                         nn.Linear(hidden_features, hidden_features),
#                                         nn.ReLU())
#             self.log_s = nn.Linear(hidden_features, in_features)
#             self.t = nn.Linear(hidden_features, in_features)
# 
#         def forward(self, x):
#             hidden = self.layers(x)
#             log_s = torch.tanh(self.log_s(hidden))
#             t = self.t(hidden)
#             return log_s, t
# 
# 
#     # In[7]:
# 
# 
#     # flow
#     flow_list = []
#     for i in range(5):
#         scale_translate_net = nn.Sequential(nn.Linear(x_dim, 256),
#                                             nn.ReLU(),
#                                             nn.Linear(256, 256),
#                                             nn.ReLU(),
#                                             nn.Linear(256, x_dim * 2))
#         flow_list.append(AffineCoupling(in_features=2,
#                                         scale_translate_net=ScaleTranslateNet(x_dim, 256),
#                                         inverse_mask=(i % 2 != 0)))
#         flow_list.append(BatchNorm1d(2))
# 
#     f = FlowList(flow_list)
# 
#     # In[8]:
# 
# 
#     # inverse transformed distribution (z -> f^-1 -> x)
#     p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"]).to(device)
#     print_latex(p)
# 
#     # In[9]:
# 
# 
#     model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-2})
#     print(model)
#     print_latex(model)
# 
#     # In[10]:
# 
# 
#     # plot training set
#     # from sklearn import datasets
# 
#     x = datasets.make_moons(n_samples=test_size, noise=0.1)[0].astype("float32")
#     noise = prior.sample(batch_n=test_size)["z"].data.cpu()
#     plot_samples(x, noise)
# 
#     # In[11]:
# 
# 
#     for epoch in range(epochs):
#         x = datasets.make_moons(n_samples=batch_size, noise=0.1)[0].astype("float32")
#         x = torch.tensor(x).to(device)
#         loss = model.train({"x": x})
# 
#         if epoch % 500 == 0:
#             print('Epoch: {} Test loss: {:.4f}'.format(epoch, loss))
# 
#             # samples
#             samples = p.sample(batch_n=test_size)["x"].data.cpu()
# 
#             # inference
#             _x = datasets.make_moons(n_samples=test_size, noise=0.1)[0].astype("float32")
#             _x = torch.tensor(_x).to(device)
#             noise = p.inference({"x": _x})["z"].data.cpu()
# 
#             plot_samples(samples, noise)
# 
#     # In[12]:
# 
# 
#     samples = p.sample(batch_n=test_size)["x"].data.cpu()
# 
#     # inference
#     _x = datasets.make_moons(n_samples=test_size, noise=0.1)[0].astype("float32")
#     _x = torch.tensor(_x).to(device)
#     noise = p.inference({"x": _x})["z"].data.cpu()
# 
#     plot_samples(samples, noise)
# 
#     # In[ ]:
# 
# 
#     # !/usr/bin/env python
#     # coding: utf-8
# 

# # Real NVP
def test_run_real_nvp():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, InverseTransformedDistribution
    from pixyz.flows import AffineCoupling, FlowList, BatchNorm1d, Shuffle, Preprocess, Reverse
    from pixyz.models import ML
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 28 * 28
    z_dim = x_dim

    # In[5]:

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_prior").to(device)

    # In[6]:

    class ScaleTranslateNet(nn.Module):
        def __init__(self, in_features, hidden_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.fc3_s = nn.Linear(hidden_features, in_features)
            self.fc3_t = nn.Linear(hidden_features, in_features)

        def forward(self, x):
            hidden = F.relu(self.fc2(F.relu(self.fc1(x))))
            log_s = torch.tanh(self.fc3_s(hidden))
            t = self.fc3_t(hidden)
            return log_s, t

    # In[7]:

    # flow
    flow_list = []
    num_block = 5

    flow_list.append(Preprocess())

    for i in range(num_block):
        flow_list.append(AffineCoupling(in_features=x_dim,
                                        scale_translate_net=ScaleTranslateNet(x_dim, 1028),
                                        inverse_mask=(i % 2 != 0)))

        flow_list.append(BatchNorm1d(x_dim))

    f = FlowList(flow_list)

    # In[8]:

    # inverse transformed distribution (z -> f^-1 -> x)
    p = InverseTransformedDistribution(prior=prior, flow=f, var=["x"]).to(device)
    print_latex(p)

    # In[9]:

    model = ML(p, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0

        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = p.forward(x, compute_jacobian=False)
            recon_batch = p.inverse(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.inverse(z_sample).view(-1, 1, 28, 28).cpu()
            return sample

    # In[13]:

    # writer = SummaryWriter()

    z_sample = torch.randn(64, z_dim).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Variational autoencoder (using the Model class)
def test_run_vae_model():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, Bernoulli
    from pixyz.losses import KullbackLeibler, Expectation as E
    from pixyz.models import Model
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 784
    z_dim = 64

    # inference model q(z|x)
    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class Generator(Bernoulli):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)

    # prior p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # In[5]:

    print(prior)
    print_latex(prior)

    # In[6]:

    print(p)
    print_latex(p)

    # In[7]:

    print(q)
    print_latex(q)

    # In[8]:

    loss = (KullbackLeibler(q, prior) - E(q, p.log_prob())).mean()
    print(loss)
    print_latex(loss)

    # In[9]:

    model = Model(loss=loss, distributions=[p, q],
                  optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[12]:

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = q.sample({"x": x}, return_all=False)
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.sample_mean({"z": z_sample}).view(-1, 1, 28, 28).cpu()
            return sample

    # In[13]:

    # writer = SummaryWriter('/runs/vae_model')

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # !/usr/bin/env python
    # coding: utf-8


# # Variational autoencoder (using the VAE class)
def test_run_vae_with_vae_class():
    # * Original paper: Auto-Encoding Variational Bayes (https://arxiv.org/pdf/1312.6114.pdf)

    # In[1]:
    # In[2]:

    # MNIST
    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.utils import print_latex

    # ## Define probability distributions
    # Prior: $p(z) = \cal N(z; \mu=0, \sigma^2=1)$  
    # Generator: $p_{\theta}(x|z) = \cal B(x; \lambda = g(z))$  
    # Inference: $q_{\phi}(z|x) = \cal N(z; \mu=f_\mu(x), \sigma^2=f_{\sigma^2}(x))$

    # In[4]:

    from pixyz.distributions import Normal, Bernoulli

    x_dim = 784
    z_dim = 64

    # inference model q(z|x)
    class Inference(Normal):
        """
        parameterizes q(z | x)
        infered z follows a Gaussian distribution with mean 'loc', variance 'scale'
        z ~ N(loc, scale)
        """

        def __init__(self):
            super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            """
            given the observation x,
            return the mean and variance of the Gaussian distritbution
            """
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class Generator(Bernoulli):
        """
        parameterizes the bernoulli(for MNIST) observation likelihood p(x | z)
        """

        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            """
            given the latent variable z,
            return the probability of Bernoulli distribution
            """
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)

    #  prior p(z)
    # z ~ N(0, 1)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # In[5]:

    print(prior)
    print_latex(prior)

    # In[6]:

    print(p)
    print_latex(p)

    # In[7]:

    print(q)
    print_latex(q)

    # ## Define VAE model using VAE Model Class
    # - https://docs.pixyz.io/en/latest/models.html#vae

    # In[8]:

    from pixyz.losses import KullbackLeibler

    # define additional loss terms for regularizing representation of latent variables
    kl = KullbackLeibler(q, prior)
    print_latex(kl)

    # In[9]:

    from pixyz.models import VAE

    model = VAE(encoder=q, decoder=p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # ## Define Train and Test loop using model

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # ## Reconstruct image and generate image

    # In[12]:

    def plot_reconstrunction(x):
        """
        reconstruct image given input observation x
        """
        with torch.no_grad():
            # infer and sampling z using inference model q `.sample()` method
            z = q.sample({"x": x}, return_all=False)

            # reconstruct image from inferred latent variable z using Generator model p `.sample_mean()` method
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            # concatenate original image and reconstructed image for comparison
            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        """
        generate new image given latent variable z
        """
        with torch.no_grad():
            # generate image from latent variable z using Generator model p `.sample_mean()` method
            sample = p.sample_mean({"z": z_sample}).view(-1, 1, 28, 28).cpu()
            return sample

    # In[13]:

    # for visualising in TensorBoard
    # writer = SummaryWriter()

    # fix latent variable z for watching generative model improvement 
    z_sample = 0.5 * torch.randn(64, z_dim).to(device)

    # set-aside observation for watching generative model improvement 
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # !/usr/bin/env python
    # coding: utf-8


# # Variational autoencoder
def test_run_vae():
    # * Original paper: Auto-Encoding Variational Bayes (https://arxiv.org/pdf/1312.6114.pdf)

    # In[1]:
    # In[2]:

    # MNIST
    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.utils import print_latex

    # ## Define probability distributions
    # Prior: $p(z) = \cal N(z; \mu=0, \sigma^2=1)$  
    # Generator: $p_{\theta}(x|z) = \cal B(x; \lambda = g(z))$  
    # Inference: $q_{\phi}(z|x) = \cal N(z; \mu=f_\mu(x), \sigma^2=f_{\sigma^2}(x))$

    # In[4]:

    from pixyz.distributions import Normal, Bernoulli

    x_dim = 784
    z_dim = 64

    # inference model q(z|x)
    class Inference(Normal):
        """
        parameterizes q(z | x)
        infered z follows a Gaussian distribution with mean 'loc', variance 'scale'
        z ~ N(loc, scale)
        """

        def __init__(self):
            super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            """
            given the observation x,
            return the mean and variance of the Gaussian distritbution
            """
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class Generator(Bernoulli):
        """
        parameterizes the bernoulli(for MNIST) observation likelihood p(x | z)
        """

        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            """
            given the latent variable z,
            return the probability of Bernoulli distribution
            """
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)

    #  prior p(z)
    # z ~ N(0, 1)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    # In[5]:

    print(prior)
    print_latex(prior)

    # In[6]:

    print(p)
    print_latex(p)

    # In[7]:

    print(q)
    print_latex(q)

    # ## Define Loss function
    # Loss function:
    # 
    # $\frac{1}{N} \sum_{i=1}^{N}\left[K L\left(q\left(z | x^{(i)}\right) \| p_{prior}(z)\right)-\mathbb{E}_{q\left(z | x^{(i)}\right)}\left[\log p\left(x^{(i)} | z\right)\right]\right]$

    # In[8]:

    from pixyz.losses import LogProb, KullbackLeibler, Expectation as E

    loss = (KullbackLeibler(q, prior) - E(q, LogProb(p))).mean()
    print_latex(loss)

    # ## Define VAE model using Model Class
    # - https://docs.pixyz.io/en/latest/models.html#model

    # In[9]:

    from pixyz.models import Model

    model = Model(loss=loss, distributions=[p, q],
                  optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # ## Define Train and Test loop using model

    # In[10]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[11]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # ## Reconstruct image and generate image

    # In[12]:

    def plot_reconstrunction(x):
        """
        reconstruct image given input observation x
        """
        with torch.no_grad():
            # infer and sampling z using inference model q `.sample()` method
            z = q.sample({"x": x}, return_all=False)

            # reconstruct image from inferred latent variable z using Generator model p `.sample_mean()` method
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            # concatenate original image and reconstructed image for comparison
            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        """
        generate new image given latent variable z
        """
        with torch.no_grad():
            # generate image from latent variable z using Generator model p `.sample_mean()` method
            sample = p.sample_mean({"z": z_sample}).view(-1, 1, 28, 28).cpu()
            return sample

    # In[13]:

    # for visualising in TensorBoard
    # writer = SummaryWriter()

    # fix latent variable z for watching generative model improvement 
    z_sample = 0.5 * torch.randn(64, z_dim).to(device)

    # set-aside observation for watching generative model improvement 
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # !/usr/bin/env python
    # coding: utf-8


# # Variational autoencoder (using the VI class)
def test_run_vi():
    # In[1]:
    # In[2]:

    # root = '../data'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Lambda(lambd=lambda x: x.view(-1))])
    # kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=True, transform=transform, download=True),
    #     shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root=root, train=False, transform=transform),
    #     shuffle=False, **kwargs)
    kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mock_mnist, shuffle=False, **kwargs)

    # In[3]:

    from pixyz.distributions import Normal, Bernoulli
    from pixyz.models import VI
    from pixyz.utils import print_latex

    # In[4]:

    x_dim = 784
    z_dim = 64

    # inference model q(z|x)
    class Inference(Normal):
        def __init__(self):
            super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")

            self.fc1 = nn.Linear(x_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc31 = nn.Linear(512, z_dim)
            self.fc32 = nn.Linear(512, z_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    # generative model p(x|z)    
    class Generator(Bernoulli):
        def __init__(self):
            super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

            self.fc1 = nn.Linear(z_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, x_dim)

        def forward(self, z):
            h = F.relu(self.fc1(z))
            h = F.relu(self.fc2(h))
            return {"probs": torch.sigmoid(self.fc3(h))}

    p = Generator().to(device)
    q = Inference().to(device)

    # prior model p(z)
    prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                   var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

    p_joint = p * prior

    # In[5]:

    print(p_joint)
    print_latex(p_joint)

    # In[6]:

    print(q)
    print_latex(q)

    # In[7]:

    model = VI(p_joint, q, optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    print(model)
    print_latex(model)

    # In[8]:

    def train(epoch):
        train_loss = 0
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            loss = model.train({"x": x})
            train_loss += loss

        train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss

    # In[9]:

    def test(epoch):
        test_loss = 0
        for x, _ in test_loader:
            x = x.to(device)
            loss = model.test({"x": x})
            test_loss += loss

        test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    # In[10]:

    def plot_reconstrunction(x):
        with torch.no_grad():
            z = q.sample({"x": x}, return_all=False)
            recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

            comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
            return comparison

    def plot_image_from_latent(z_sample):
        with torch.no_grad():
            sample = p.sample_mean({"z": z_sample}).view(-1, 1, 28, 28).cpu()
            return sample

    # In[11]:

    # writer = SummaryWriter()

    z_sample = 0.5 * torch.randn(64, z_dim).to(device)
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)

        recon = plot_reconstrunction(_x[:8])
        sample = plot_image_from_latent(z_sample)

    #     writer.add_scalar('train_loss', train_loss.item(), epoch)
    #     writer.add_scalar('test_loss', test_loss.item(), epoch)
    # 
    #     writer.add_images('Image_from_latent', sample, epoch)
    #     writer.add_images('Image_reconstrunction', recon, epoch)
    # 
    # writer.close()

    # In[ ]:
