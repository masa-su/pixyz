{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Action Conditional Deep Markov Model using cartpole dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ubuntu/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py:1067: UserWarning: Duplicate key in file \"/home/ubuntu/.config/matplotlib/matplotlibrc\", line #2\n",
      "  (fname, cnt))\n",
      "/home/ubuntu/anaconda3/lib/python3.6/site-packages/matplotlib/__init__.py:1067: UserWarning: Duplicate key in file \"/home/ubuntu/.config/matplotlib/matplotlibrc\", line #3\n",
      "  (fname, cnt))\n"
     ]
    }
   ],
   "source": [
    "from tqdm import tqdm\n",
    "\n",
    "import torch\n",
    "from torch import optim\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torchvision import transforms, datasets\n",
    "from tensorboardX import SummaryWriter\n",
    "import numpy as np\n",
    "\n",
    "from utils import DMMDataset, imshow, postprocess\n",
    "from torch.utils.data import DataLoader\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "seed = 1\n",
    "torch.manual_seed(seed)\n",
    "if torch.cuda.is_available():\n",
    "    device = \"cuda\"\n",
    "else:\n",
    "    device = \"cpu\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prepare Dataset \n",
    "you have to run prepare_cartpole_dataset.py or download from :  \n",
    "https://drive.google.com/drive/folders/1w_97RLFS--CpdUCNw1C-3yPLhceZxkO2?usp=sharing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([30, 3, 28, 28])\n"
     ]
    }
   ],
   "source": [
    "batch_size = 256\n",
    "train_loader = DataLoader(DMMDataset(), batch_size=batch_size, shuffle=True, drop_last=True)\n",
    "# test_loader = DataLoader(DMMTestDataset(), batch_size=batch_size, shuffle=False, drop_last=True)\n",
    "\n",
    "_x = iter(train_loader).next()\n",
    "print(_x['episode_frames'][0][0:30].shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([30, 3, 28, 28])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x864 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.]])\n"
     ]
    }
   ],
   "source": [
    "imshow(postprocess(_x['episode_frames'][0][0:30]))\n",
    "\n",
    "# 0: Push cart to the left\n",
    "# 1:Push cart to the right\n",
    "print(_x['actions'][0][0:30])\n",
    "\n",
    "# for more details about actions: https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/envs/classic_control/cartpole.py#L37\n",
    "# for more details about CartPole-v1: https://gym.openai.com/envs/CartPole-v1/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pixyz.utils import print_latex\n",
    "from pixyz.distributions import Bernoulli, Normal, Deterministic\n",
    "\n",
    "\n",
    "h_dim = 32\n",
    "hidden_dim = 32\n",
    "z_dim = 16\n",
    "t_max = 30\n",
    "u_dim = 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deep Markov Model\n",
    "* Original paper: Structured Inference Networks for Nonlinear State Space Models (https://arxiv.org/abs/1609.09869)\n",
    "* Original code: https://github.com/clinicalml/dmm\n",
    "\n",
    "\n",
    "Prior(Transition model): $p_{\\theta}(z_{t} | z_{t-1}, u) =  \\cal{N}(\\mu = f_{prior_\\mu}(z_{t-1}, u), \\sigma^2 = f_{prior_\\sigma^2}(z_{t-1}, u)$    \n",
    "Generator(Emission): $p_{\\theta}(x | z)=\\mathscr{B}\\left(x ; \\lambda=g_{x}(z)\\right)$  \n",
    "\n",
    "RNN: $p(h) = RNN(x)$  \n",
    "Inference(Combiner): $p_{\\phi}(z | h, z_{t-1}, u) = \\cal{N}(\\mu = f_{\\mu}(h, z_{t-1}, u), \\sigma^2 = f_{\\sigma^2}(h, z_{t-1}, u)$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define probability distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# RNN\n",
    "class RNN(Deterministic):\n",
    "    \"\"\"\n",
    "    h = RNN(x)\n",
    "    Given observed x, RNN output hidden state\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(RNN, self).__init__(var=[\"h\"], cond_var=[\"x\"])\n",
    "        \n",
    "        # 28x28x3 → 32\n",
    "        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)\n",
    "        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)\n",
    "        self.fc1 = nn.Linear(128*7*7, 256)\n",
    "        self.fc2 = nn.Linear(256, 32)\n",
    "        \n",
    "        self.rnn = nn.GRU(32, h_dim, bidirectional=True)\n",
    "        self.h0 = nn.Parameter(torch.zeros(2, 1, self.rnn.hidden_size))\n",
    "        self.hidden_size = self.rnn.hidden_size\n",
    "        \n",
    "    def forward(self, x):\n",
    "        \n",
    "        h0 = self.h0.expand(2, x.size(1), self.rnn.hidden_size).contiguous()\n",
    "        x = x.reshape(-1, 3, 28, 28)      # Nx3x28x28\n",
    "\n",
    "        h = F.relu(self.conv1(x))         # Nx64x14x14\n",
    "        h = F.relu(self.conv2(h))         # Nx128x7x7\n",
    "        h = h.view(h.shape[0], 128*7*7)   # Nx128*7*7\n",
    "        h = F.relu(self.fc1(h))           # Nx256\n",
    "        h = F.relu(self.fc2(h))           # Nx32\n",
    "        h = h.reshape(30, -1, 32)         # 30x128x32\n",
    "\n",
    "        h, _ = self.rnn(h, h0)            # 30x128x32, 1x128x32\n",
    "        return {\"h\": h}\n",
    "\n",
    "\n",
    "# Emission p(x_t | z_t)\n",
    "class Generator(Normal):\n",
    "    \"\"\"\n",
    "    Given the latent z at time step t, return the vector of\n",
    "    probabilities that parameterizes the bernlulli distribution p(x_t | z_t)\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(Generator, self).__init__(var=[\"x\"], cond_var=[\"z\"])\n",
    "        self.fc1 = nn.Linear(z_dim, 256)\n",
    "        self.fc2 = nn.Linear(256, 128*7*7)\n",
    "        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)\n",
    "        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)        \n",
    "\n",
    "    def forward(self, z):\n",
    "        h = F.relu(self.fc1(z))\n",
    "        h = F.relu(self.fc2(h))\n",
    "        h = h.view(h.shape[0], 128, 7, 7) # 128*7*7\n",
    "        h = F.relu(self.conv1(h))         # 64x14x14\n",
    "        generated_x = self.conv2(h)                 # 3x28x28\n",
    "        return {\"loc\": generated_x, \"scale\": 1.0}\n",
    "\n",
    "\n",
    "class Inference(Normal):\n",
    "    \"\"\"\n",
    "    given the latent z at time step t-1, the hidden state of the RNN h(x_{0:T} and u\n",
    "    return the loc and scale vectors that\n",
    "    parameterize the gaussian distribution q(z_t | z_{t-1}, x_{t:T}, u)\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(Inference, self).__init__(var=[\"z\"], cond_var=[\"h\", \"z_prev\", \"u\"])\n",
    "        self.fc1 = nn.Linear(z_dim+u_dim, h_dim*2)\n",
    "        self.fc21 = nn.Linear(h_dim*2, z_dim)\n",
    "        self.fc22 = nn.Linear(h_dim*2, z_dim)\n",
    "\n",
    "        \n",
    "    def forward(self, h, z_prev, u):\n",
    "        feature = torch.cat((z_prev, u), 1)\n",
    "        h_z = torch.tanh(self.fc1(feature))\n",
    "        h = 0.5 * (h + h_z)\n",
    "        return {\"loc\": self.fc21(h), \"scale\": F.softplus(self.fc22(h))}\n",
    "\n",
    "\n",
    "class Prior(Normal):\n",
    "    \"\"\"\n",
    "    Given the latent variable at the time step t-1 and u,\n",
    "    return the mean and scale vectors that parameterize the\n",
    "    gaussian distribution p(z_t | z_{t-1}, u)\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(Prior, self).__init__(var=[\"z\"], cond_var=[\"z_prev\", \"u\"])\n",
    "        self.fc1 = nn.Linear(z_dim+u_dim, hidden_dim)\n",
    "        self.fc21 = nn.Linear(hidden_dim, z_dim)\n",
    "        self.fc22 = nn.Linear(hidden_dim, z_dim)\n",
    "        \n",
    "    def forward(self, z_prev, u):\n",
    "        feature = torch.cat((z_prev, u), 1)\n",
    "        h = F.relu(self.fc1(feature))\n",
    "        return {\"loc\": self.fc21(h), \"scale\": F.softplus(self.fc22(h))}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/latex": [
       "$$p(x,z|z_{prev},u) = p(x|z)p(z|z_{prev},u)$$"
      ],
      "text/plain": [
       "<IPython.core.display.Math object>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prior = Prior().to(device)\n",
    "encoder = Inference().to(device)\n",
    "decoder = Generator().to(device)\n",
    "rnn = RNN().to(device)\n",
    "generate_from_prior = prior * decoder\n",
    "\n",
    "print_latex(generate_from_prior)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pixyz.losses import KullbackLeibler\n",
    "from pixyz.losses import Expectation as E\n",
    "from pixyz.losses import LogProb\n",
    "from pixyz.losses import IterativeLoss\n",
    "\n",
    "step_loss = - E(encoder, LogProb(decoder)) + KullbackLeibler(encoder, prior)\n",
    "\n",
    "# IterativeLoss: https://docs.pixyz.io/en/latest/losses.html#pixyz.losses.IterativeLoss\n",
    "_loss = IterativeLoss(step_loss, max_iter=t_max, \n",
    "                      series_var=[\"x\", \"h\", \"u\"], update_value={\"z\": \"z_prev\"})\n",
    "loss = E(rnn, _loss).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Distributions (for training): \n",
      "  p(h|x), p(z|h,z_{prev},u), p(x|z), p(z|z_{prev},u) \n",
      "Loss function: \n",
      "  mean \\left(\\mathbb{E}_{p(h|x)} \\left[\\sum_{t=1}^{30} \\left(D_{KL} \\left[p(z|h,z_{prev},u)||p(z|z_{prev},u) \\right] - \\mathbb{E}_{p(z|h,z_{prev},u)} \\left[\\log p(x|z) \\right]\\right) \\right] \\right) \n",
      "Optimizer: \n",
      "  RMSprop (\n",
      "  Parameter Group 0\n",
      "      alpha: 0.99\n",
      "      centered: False\n",
      "      eps: 1e-08\n",
      "      lr: 0.0005\n",
      "      momentum: 0\n",
      "      weight_decay: 0\n",
      "  )\n"
     ]
    },
    {
     "data": {
      "text/latex": [
       "$$mean \\left(\\mathbb{E}_{p(h|x)} \\left[\\sum_{t=1}^{30} \\left(D_{KL} \\left[p(z|h,z_{prev},u)||p(z|z_{prev},u) \\right] - \\mathbb{E}_{p(z|h,z_{prev},u)} \\left[\\log p(x|z) \\right]\\right) \\right] \\right)$$"
      ],
      "text/plain": [
       "<IPython.core.display.Math object>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from pixyz.models import Model\n",
    "\n",
    "dmm = Model(loss, distributions=[rnn, encoder, decoder, prior], \n",
    "            optimizer=optim.RMSprop, optimizer_params={\"lr\": 5e-4}, clip_grad_value=10)\n",
    "\n",
    "print(dmm)\n",
    "print_latex(dmm)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sampling code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def data_loop(epoch, loader, model, device, train_mode=False):\n",
    "    mean_loss = 0\n",
    "    for data in loader:\n",
    "        x = data['episode_frames'].to(device) # 256,30,3,28,28\n",
    "        u = data['actions'].to(device) # 256,30,1\n",
    "        batch_size = x.size()[0]\n",
    "        x = x.transpose(0, 1) # 30,256,3,28,28\n",
    "        u = u.transpose(0, 1) # 30,256,1\n",
    "        z_prev = torch.zeros(batch_size, z_dim).to(device)\n",
    "        if train_mode:\n",
    "            mean_loss += model.train({'x': x, 'z_prev': z_prev, 'u': u}).item() * batch_size\n",
    "        else:\n",
    "            mean_loss += model.test({'x': x, 'z_prev': z_prev, 'u': u}).item() * batch_size\n",
    "    mean_loss /= len(loader.dataset)\n",
    "    if train_mode:\n",
    "        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))\n",
    "    else:\n",
    "        print('Test loss: {:.4f}'.format(mean_loss))\n",
    "    return mean_loss\n",
    "\n",
    "_data = iter(train_loader).next()\n",
    "_u = _data['actions'].to(device) # 256,30,1\n",
    "_u = _u.transpose(0, 1)          # 30,256,1\n",
    "\n",
    "def plot_video_from_latent(batch_size):\n",
    "    x = []\n",
    "    z_prev = torch.zeros(batch_size, z_dim).to(device)\n",
    "    for step in range(t_max):\n",
    "        samples = generate_from_prior.sample({'z_prev': z_prev, 'u': _u[step]})\n",
    "        x_t = decoder.sample_mean({\"z\": samples[\"z\"]})\n",
    "        z_prev = samples[\"z\"]\n",
    "        x.append(x_t[None, :])\n",
    "    x = torch.cat(x, dim=0).transpose(0, 1)\n",
    "    return x"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  0%|          | 0/200 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1 Train loss: 135122.8520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  0%|          | 1/200 [00:04<13:35,  4.10s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 2 Train loss: 90253.1340\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  1%|          | 2/200 [00:08<13:16,  4.02s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 3 Train loss: 81096.6820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  2%|▏         | 3/200 [00:11<13:07,  4.00s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 4 Train loss: 77355.0100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  2%|▏         | 4/200 [00:15<12:55,  3.96s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 5 Train loss: 73845.4700\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  2%|▎         | 5/200 [00:19<12:48,  3.94s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 6 Train loss: 72047.6200\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  3%|▎         | 6/200 [00:23<12:43,  3.94s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 7 Train loss: 69971.1520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▎         | 7/200 [00:27<12:37,  3.93s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 8 Train loss: 69165.3120\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▍         | 8/200 [00:31<12:31,  3.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 9 Train loss: 68188.5620\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▍         | 9/200 [00:35<12:28,  3.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 10 Train loss: 67598.2300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  5%|▌         | 10/200 [00:39<12:23,  3.91s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 11 Train loss: 67193.7760\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  6%|▌         | 11/200 [00:43<12:19,  3.91s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 12 Train loss: 66844.7420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  6%|▌         | 12/200 [00:47<12:16,  3.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 13 Train loss: 66364.0060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  6%|▋         | 13/200 [00:50<12:11,  3.91s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 14 Train loss: 66334.7240\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  7%|▋         | 14/200 [00:54<12:09,  3.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 15 Train loss: 65878.0960\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|▊         | 15/200 [00:58<12:04,  3.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 16 Train loss: 65880.0440\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|▊         | 16/200 [01:02<11:59,  3.91s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 17 Train loss: 65475.5400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|▊         | 17/200 [01:06<11:54,  3.90s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 18 Train loss: 65414.8520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  9%|▉         | 18/200 [01:10<11:49,  3.90s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 19 Train loss: 64852.2780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 10%|▉         | 19/200 [01:14<11:45,  3.90s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 20 Train loss: 64408.8360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 10%|█         | 20/200 [01:17<11:40,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 21 Train loss: 63974.0380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 10%|█         | 21/200 [01:21<11:35,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 22 Train loss: 63630.9260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 11%|█         | 22/200 [01:25<11:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 23 Train loss: 63318.6640\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 12%|█▏        | 23/200 [01:29<11:26,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 24 Train loss: 63170.1460\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 12%|█▏        | 24/200 [01:33<11:22,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 25 Train loss: 62920.6320\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 12%|█▎        | 25/200 [01:36<11:18,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 26 Train loss: 62751.3140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 13%|█▎        | 26/200 [01:40<11:14,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 27 Train loss: 62510.9520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 14%|█▎        | 27/200 [01:44<11:10,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 28 Train loss: 62588.4360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 14%|█▍        | 28/200 [01:48<11:06,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 29 Train loss: 61929.8660\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 14%|█▍        | 29/200 [01:52<11:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 30 Train loss: 61316.8060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 15%|█▌        | 30/200 [01:56<10:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 31 Train loss: 60486.2660\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█▌        | 31/200 [02:00<10:54,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 32 Train loss: 60001.6220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█▌        | 32/200 [02:04<10:51,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 33 Train loss: 59460.8240\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█▋        | 33/200 [02:07<10:47,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 34 Train loss: 59289.5080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 17%|█▋        | 34/200 [02:11<10:43,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 35 Train loss: 59073.3060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 18%|█▊        | 35/200 [02:15<10:39,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 36 Train loss: 58960.7100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 18%|█▊        | 36/200 [02:19<10:35,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 37 Train loss: 58642.3800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 18%|█▊        | 37/200 [02:23<10:30,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 38 Train loss: 58429.8240\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 19%|█▉        | 38/200 [02:27<10:26,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 39 Train loss: 58304.1100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 20%|█▉        | 39/200 [02:30<10:22,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 40 Train loss: 58185.2540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 20%|██        | 40/200 [02:34<10:18,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 41 Train loss: 57996.4080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 20%|██        | 41/200 [02:38<10:14,  3.86s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 42 Train loss: 57931.2140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 21%|██        | 42/200 [02:42<10:10,  3.86s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 43 Train loss: 57737.4840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 22%|██▏       | 43/200 [02:45<10:05,  3.86s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 44 Train loss: 57706.4700\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 22%|██▏       | 44/200 [02:49<10:01,  3.86s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 45 Train loss: 57439.6060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 22%|██▎       | 45/200 [02:53<09:57,  3.86s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 46 Train loss: 57463.6020\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 23%|██▎       | 46/200 [02:57<09:53,  3.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 47 Train loss: 57285.5520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 24%|██▎       | 47/200 [03:01<09:49,  3.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 48 Train loss: 57247.4760\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 24%|██▍       | 48/200 [03:04<09:45,  3.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 49 Train loss: 57086.2280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 24%|██▍       | 49/200 [03:08<09:41,  3.85s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 50 Train loss: 56966.7360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x216 with 30 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 25%|██▌       | 50/200 [03:13<09:41,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 51 Train loss: 56824.1820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 26%|██▌       | 51/200 [03:17<09:37,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 52 Train loss: 56812.5080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 26%|██▌       | 52/200 [03:21<09:34,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 53 Train loss: 56666.7160\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 26%|██▋       | 53/200 [03:25<09:30,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 54 Train loss: 56668.7020\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 27%|██▋       | 54/200 [03:29<09:26,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 55 Train loss: 56497.9160\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 28%|██▊       | 55/200 [03:33<09:22,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 56 Train loss: 56479.9800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 28%|██▊       | 56/200 [03:37<09:18,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 57 Train loss: 56297.6360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 28%|██▊       | 57/200 [03:41<09:14,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 58 Train loss: 56295.1740\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 29%|██▉       | 58/200 [03:44<09:10,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 59 Train loss: 56162.6200\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 30%|██▉       | 59/200 [03:48<09:06,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 60 Train loss: 56182.1080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 30%|███       | 60/200 [03:52<09:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 61 Train loss: 56106.6000\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 30%|███       | 61/200 [03:56<08:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 62 Train loss: 56055.0940\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 31%|███       | 62/200 [04:00<08:54,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 63 Train loss: 55902.8800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 32%|███▏      | 63/200 [04:04<08:51,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 64 Train loss: 55918.1920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 32%|███▏      | 64/200 [04:08<08:47,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 65 Train loss: 55805.0120\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 32%|███▎      | 65/200 [04:11<08:43,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 66 Train loss: 55797.4460\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 33%|███▎      | 66/200 [04:15<08:39,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 67 Train loss: 55713.8000\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 34%|███▎      | 67/200 [04:19<08:35,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 68 Train loss: 55664.8560\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 34%|███▍      | 68/200 [04:23<08:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 69 Train loss: 55576.3620\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 34%|███▍      | 69/200 [04:27<08:27,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 70 Train loss: 55548.3420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 35%|███▌      | 70/200 [04:31<08:23,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 71 Train loss: 55429.2280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 36%|███▌      | 71/200 [04:35<08:20,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 72 Train loss: 55480.1360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 36%|███▌      | 72/200 [04:39<08:16,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 73 Train loss: 55372.8140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 36%|███▋      | 73/200 [04:43<08:12,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 74 Train loss: 55322.9160\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 37%|███▋      | 74/200 [04:46<08:08,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 75 Train loss: 55238.1540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 38%|███▊      | 75/200 [04:50<08:04,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 76 Train loss: 55280.2400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 38%|███▊      | 76/200 [04:54<08:00,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 77 Train loss: 55179.4380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 38%|███▊      | 77/200 [04:58<07:57,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 78 Train loss: 55209.4780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 39%|███▉      | 78/200 [05:02<07:53,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 79 Train loss: 55079.0920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 40%|███▉      | 79/200 [05:06<07:49,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 80 Train loss: 55045.5060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 40%|████      | 80/200 [05:10<07:45,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 81 Train loss: 55030.6360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 40%|████      | 81/200 [05:13<07:41,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 82 Train loss: 55005.8220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 41%|████      | 82/200 [05:17<07:37,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 83 Train loss: 54929.9360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 42%|████▏     | 83/200 [05:21<07:33,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 84 Train loss: 54969.1100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 42%|████▏     | 84/200 [05:25<07:29,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 85 Train loss: 54869.3160\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 42%|████▎     | 85/200 [05:29<07:25,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 86 Train loss: 54861.0800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 43%|████▎     | 86/200 [05:33<07:21,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 87 Train loss: 54751.7260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 44%|████▎     | 87/200 [05:37<07:17,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 88 Train loss: 54815.0680\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 44%|████▍     | 88/200 [05:41<07:14,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 89 Train loss: 54739.2640\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 44%|████▍     | 89/200 [05:44<07:10,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 90 Train loss: 54757.9560\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 45%|████▌     | 90/200 [05:48<07:06,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 91 Train loss: 54705.2020\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 46%|████▌     | 91/200 [05:52<07:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 92 Train loss: 54654.3220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 46%|████▌     | 92/200 [05:56<06:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 93 Train loss: 54617.0460\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 46%|████▋     | 93/200 [06:00<06:54,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 94 Train loss: 54623.8140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 47%|████▋     | 94/200 [06:04<06:50,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 95 Train loss: 54542.4060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 48%|████▊     | 95/200 [06:08<06:46,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 96 Train loss: 54619.6100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 48%|████▊     | 96/200 [06:12<06:43,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 97 Train loss: 54504.6980\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 48%|████▊     | 97/200 [06:16<06:39,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 98 Train loss: 54569.3900\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 49%|████▉     | 98/200 [06:19<06:35,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 99 Train loss: 54432.7600\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 50%|████▉     | 99/200 [06:23<06:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 100 Train loss: 54470.5920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x216 with 30 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 50%|█████     | 100/200 [06:29<06:29,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 101 Train loss: 54403.0380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 50%|█████     | 101/200 [06:33<06:25,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 102 Train loss: 54435.0420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 51%|█████     | 102/200 [06:36<06:21,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 103 Train loss: 54357.4960\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 52%|█████▏    | 103/200 [06:40<06:17,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 104 Train loss: 54368.0540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 52%|█████▏    | 104/200 [06:44<06:13,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 105 Train loss: 54305.8840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 52%|█████▎    | 105/200 [06:48<06:09,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 106 Train loss: 54326.0220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 53%|█████▎    | 106/200 [06:52<06:05,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 107 Train loss: 54248.3700\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 54%|█████▎    | 107/200 [06:55<06:01,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 108 Train loss: 54299.8040\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 54%|█████▍    | 108/200 [06:59<05:57,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 109 Train loss: 54227.2680\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 55%|█████▍    | 109/200 [07:03<05:53,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 110 Train loss: 54208.4380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 55%|█████▌    | 110/200 [07:07<05:49,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 111 Train loss: 54137.3380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 56%|█████▌    | 111/200 [07:11<05:45,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 112 Train loss: 54148.8340\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 56%|█████▌    | 112/200 [07:15<05:41,  3.89s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 113 Train loss: 54160.9420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 56%|█████▋    | 113/200 [07:19<05:37,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 114 Train loss: 54132.4100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 57%|█████▋    | 114/200 [07:22<05:34,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 115 Train loss: 54045.9820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 57%|█████▊    | 115/200 [07:26<05:30,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 116 Train loss: 54099.1520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 58%|█████▊    | 116/200 [07:30<05:26,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 117 Train loss: 54069.4480\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 58%|█████▊    | 117/200 [07:34<05:22,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 118 Train loss: 54081.5580\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 59%|█████▉    | 118/200 [07:38<05:18,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 119 Train loss: 54045.7140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 60%|█████▉    | 119/200 [07:41<05:14,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 120 Train loss: 54002.5400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 60%|██████    | 120/200 [07:45<05:10,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 121 Train loss: 53965.1260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 60%|██████    | 121/200 [07:49<05:06,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 122 Train loss: 53992.2380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 61%|██████    | 122/200 [07:53<05:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 123 Train loss: 53972.8280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 62%|██████▏   | 123/200 [07:57<04:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 124 Train loss: 53969.1620\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 62%|██████▏   | 124/200 [08:01<04:54,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 125 Train loss: 53927.5200\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 62%|██████▎   | 125/200 [08:04<04:50,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 126 Train loss: 53964.3320\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 63%|██████▎   | 126/200 [08:08<04:47,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 127 Train loss: 53869.5820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 64%|██████▎   | 127/200 [08:12<04:43,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 128 Train loss: 53871.5780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 64%|██████▍   | 128/200 [08:16<04:39,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 129 Train loss: 53859.7800\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 64%|██████▍   | 129/200 [08:20<04:35,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 130 Train loss: 53908.0260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 65%|██████▌   | 130/200 [08:24<04:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 131 Train loss: 53837.9960\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 66%|██████▌   | 131/200 [08:27<04:27,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 132 Train loss: 53871.7580\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 66%|██████▌   | 132/200 [08:31<04:23,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 133 Train loss: 53799.4480\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 66%|██████▋   | 133/200 [08:35<04:19,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 134 Train loss: 53817.8300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 67%|██████▋   | 134/200 [08:39<04:15,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 135 Train loss: 53753.9780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 68%|██████▊   | 135/200 [08:43<04:11,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 136 Train loss: 53785.4380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 68%|██████▊   | 136/200 [08:47<04:08,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 137 Train loss: 53714.4720\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 68%|██████▊   | 137/200 [08:50<04:04,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 138 Train loss: 53829.2060\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 69%|██████▉   | 138/200 [08:54<04:00,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 139 Train loss: 53734.8520\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 70%|██████▉   | 139/200 [08:58<03:56,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 140 Train loss: 53774.9560\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 70%|███████   | 140/200 [09:02<03:52,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 141 Train loss: 53710.4300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 70%|███████   | 141/200 [09:06<03:48,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 142 Train loss: 53724.4580\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 71%|███████   | 142/200 [09:09<03:44,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 143 Train loss: 53684.8880\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 72%|███████▏  | 143/200 [09:13<03:40,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 144 Train loss: 53695.5900\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 72%|███████▏  | 144/200 [09:17<03:36,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 145 Train loss: 53645.0780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 72%|███████▎  | 145/200 [09:21<03:32,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 146 Train loss: 53627.6320\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 73%|███████▎  | 146/200 [09:25<03:29,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 147 Train loss: 53627.0840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 74%|███████▎  | 147/200 [09:29<03:25,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 148 Train loss: 53624.9380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 74%|███████▍  | 148/200 [09:32<03:21,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 149 Train loss: 53569.0120\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 74%|███████▍  | 149/200 [09:36<03:17,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 150 Train loss: 53564.5840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x216 with 30 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 75%|███████▌  | 150/200 [09:41<03:13,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 151 Train loss: 53580.5680\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 76%|███████▌  | 151/200 [09:45<03:10,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 152 Train loss: 53582.3500\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 76%|███████▌  | 152/200 [09:49<03:06,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 153 Train loss: 53538.7780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 76%|███████▋  | 153/200 [09:53<03:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 154 Train loss: 53602.6540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 77%|███████▋  | 154/200 [09:57<02:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 155 Train loss: 53534.8760\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 78%|███████▊  | 155/200 [10:00<02:54,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 156 Train loss: 53510.2260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 78%|███████▊  | 156/200 [10:04<02:50,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 157 Train loss: 53473.8100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 78%|███████▊  | 157/200 [10:08<02:46,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 158 Train loss: 53535.6280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 79%|███████▉  | 158/200 [10:12<02:42,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 159 Train loss: 53473.0280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 80%|███████▉  | 159/200 [10:16<02:38,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 160 Train loss: 53514.6540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 80%|████████  | 160/200 [10:19<02:34,  3.87s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 161 Train loss: 53455.0380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 80%|████████  | 161/200 [10:23<02:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 162 Train loss: 53475.0180\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 81%|████████  | 162/200 [10:27<02:27,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 163 Train loss: 53475.7480\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 82%|████████▏ | 163/200 [10:31<02:23,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 164 Train loss: 53454.6420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 82%|████████▏ | 164/200 [10:35<02:19,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 165 Train loss: 53427.2280\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 82%|████████▎ | 165/200 [10:39<02:15,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 166 Train loss: 53426.8360\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 83%|████████▎ | 166/200 [10:43<02:11,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 167 Train loss: 53390.6420\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 84%|████████▎ | 167/200 [10:47<02:07,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 168 Train loss: 53409.6440\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 84%|████████▍ | 168/200 [10:51<02:04,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 169 Train loss: 53358.5380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 84%|████████▍ | 169/200 [10:55<02:00,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 170 Train loss: 53430.7460\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 85%|████████▌ | 170/200 [10:59<01:56,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 171 Train loss: 53373.2400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 86%|████████▌ | 171/200 [11:03<01:52,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 172 Train loss: 53368.7840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 86%|████████▌ | 172/200 [11:06<01:48,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 173 Train loss: 53353.5260\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 86%|████████▋ | 173/200 [11:10<01:44,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 174 Train loss: 53360.5540\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 87%|████████▋ | 174/200 [11:14<01:40,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 175 Train loss: 53351.1160\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 88%|████████▊ | 175/200 [11:18<01:36,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 176 Train loss: 53340.8240\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 88%|████████▊ | 176/200 [11:22<01:33,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 177 Train loss: 53299.8320\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 88%|████████▊ | 177/200 [11:26<01:29,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 178 Train loss: 53322.5840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 89%|████████▉ | 178/200 [11:30<01:25,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 179 Train loss: 53275.7780\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 90%|████████▉ | 179/200 [11:34<01:21,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 180 Train loss: 53316.4840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 90%|█████████ | 180/200 [11:38<01:17,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 181 Train loss: 53294.6960\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 90%|█████████ | 181/200 [11:42<01:13,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 182 Train loss: 53332.1500\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 91%|█████████ | 182/200 [11:46<01:09,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 183 Train loss: 53249.3300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 92%|█████████▏| 183/200 [11:49<01:05,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 184 Train loss: 53328.4840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 92%|█████████▏| 184/200 [11:53<01:02,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 185 Train loss: 53218.7080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 92%|█████████▎| 185/200 [11:57<00:58,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 186 Train loss: 53264.6480\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 93%|█████████▎| 186/200 [12:01<00:54,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 187 Train loss: 53213.9380\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 94%|█████████▎| 187/200 [12:05<00:50,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 188 Train loss: 53233.1180\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 94%|█████████▍| 188/200 [12:09<00:46,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 189 Train loss: 53181.2920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 94%|█████████▍| 189/200 [12:13<00:42,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 190 Train loss: 53215.2880\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 95%|█████████▌| 190/200 [12:17<00:38,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 191 Train loss: 53185.2100\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 96%|█████████▌| 191/200 [12:21<00:34,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 192 Train loss: 53215.6580\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 96%|█████████▌| 192/200 [12:24<00:31,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 193 Train loss: 53168.9000\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 96%|█████████▋| 193/200 [12:28<00:27,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 194 Train loss: 53181.8680\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 97%|█████████▋| 194/200 [12:32<00:23,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 195 Train loss: 53125.8080\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 98%|█████████▊| 195/200 [12:36<00:19,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 196 Train loss: 53162.7740\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 98%|█████████▊| 196/200 [12:40<00:15,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 197 Train loss: 53161.7220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 98%|█████████▊| 197/200 [12:44<00:11,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 198 Train loss: 53189.5240\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 99%|█████████▉| 198/200 [12:47<00:07,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 199 Train loss: 53112.2320\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "100%|█████████▉| 199/200 [12:51<00:03,  3.88s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 200 Train loss: 53137.6600\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n",
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x216 with 30 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 200/200 [12:56<00:00,  3.88s/it]\n"
     ]
    }
   ],
   "source": [
    "epochs = 200\n",
    "for epoch in tqdm(range(1, epochs + 1)):\n",
    "    train_loss = data_loop(epoch, train_loader, dmm, device, train_mode=True)\n",
    "    sample = plot_video_from_latent(batch_size)\n",
    "    if epoch % 50 == 0:\n",
    "        plt.figure(figsize=(10,3))\n",
    "        for i in range(30):\n",
    "            plt.subplot(3,10,i+1)\n",
    "            plt.imshow(sample[0][i].cpu().detach().numpy().astype(np.float).reshape(3,28,28).transpose(1,2,0))\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([30, 3, 28, 28])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x864 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "imshow(postprocess(sample[0].cpu().detach()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([30, 3, 28, 28])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x864 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.],\n",
      "        [1.],\n",
      "        [0.],\n",
      "        [0.]])\n"
     ]
    }
   ],
   "source": [
    "imshow(postprocess(_data[\"episode_frames\"][0]))\n",
    "print(_data[\"actions\"][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
