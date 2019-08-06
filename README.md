# Pixyz: A library for developing deep generative models

<img src="https://user-images.githubusercontent.com/11865486/58864169-3706a980-86ef-11e9-82f4-18bb0275271b.png" width="800px">

[![pypi](https://img.shields.io/pypi/v/pixyz.svg)](https://pypi.python.org/pypi/pixyz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/masa-su/pixyz)
[![Pytorch Version](https://img.shields.io/badge/pytorch-1.0-yellow.svg)](https://github.com/masa-su/pixyz)
[![Read the Docs](https://readthedocs.org/projects/pixyz/badge/?version=latest)](http://docs.pixyz.io)
[![TravisCI](https://travis-ci.org/masa-su/pixyz.svg?branch=master)](https://github.com/masa-su/pixyz)

[Docs](https://docs.pixyz.io) | [Examples](https://github.com/masa-su/pixyz/tree/master/examples) | [Pixyzoo](https://github.com/masa-su/pixyzoo)

- [What is Pixyz?](#what-is-pixyz)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [More information](#more-information)
- [Acknowledgements](#acknowledgements)

## What is Pixyz?
[**Pixyz**](https://github.com/masa-su/pixyz) is a high-level deep generative modeling library, based on [PyTorch](https://pytorch.org/). It is developed with a focus on enabling easy implementation of various deep generative models.

Recently, many papers about deep generative models have been published. However, its reproduction becomes a hard task, for both specialists and practitioners, because such recent models become more complex and there are no unified tools that bridge mathematical formulation of them and implementation. The vision of our library is to enable both specialists and practitioners to implement such complex deep generative models by **just as if writing the formulas provided in these papers**.

Our library supports the following deep generative models.

* Explicit models (likelihood-based)
  * Variational autoencoders (variational inference)
  * Flow-based models
  * Autoregressive generative models (note: not implemented yet)
* Implicit models
  * Generative adversarial networks
  
Moreover, Pixyz enables you to implement these different models **in the same framework** and **in combination with each other**.

The overview of Pixyz is as follows. Each API will be discussed below.
<img src="https://user-images.githubusercontent.com/11865486/58321994-a3b1b680-7e5a-11e9-89dd-334086a89525.png" width="600px">

**Note**: Since this library is under development, there are possibilities to have some bugs.

## Installation

Pixyz can be installed by using `pip`.
```
$ pip install pixyz
```

If installing from source code, execute the following commands.
```
$ git clone https://github.com/masa-su/pixyz.git
$ pip install -e pixyz
```

## Quick Start

Here, we consider to implement a variational auto-encoder (VAE) which is one of the most well-known deep generative models. VAE is composed of a inference model
<img src="https://latex.codecogs.com/gif.latex?q_{\phi}(z|x)" />
and a generative model
<img src="https://latex.codecogs.com/gif.latex?p_{\theta}(x,z)=p_{\theta}(x|z)p(z)" />
 , each of which is defined by DNN, and this loss function (negative ELBO) is as follows.

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(x;\phi,\theta)=-E_{q_{\phi}(z|x)}\left[\log{p_{\theta}(x|z)}\right]+D_{KL}\left[q_{\phi}(z|x)||p_{prior}(z)\right]" /> (1)

In Pixyz, deep generative models are implemented in the following three steps:
1. [Define distributions(Distribution API）](#1-define-distributionsdistribution-api)
2. [Set the loss function of a model(Loss API)](#2-set-the-loss-function-of-a-modelloss-api)
3. [Train the model(Model API)](#3-train-the-modelmodel-api)

### 1. Define distributions(Distribution API)
First, we need to define two distributions (
<img src="https://latex.codecogs.com/gif.latex?q_{\phi}(z|x)" />
,
<img src="https://latex.codecogs.com/gif.latex?p_{\theta}(x|z)" />
) with DNNs. In Pixyz, you can do this by building DNN modules just as you do in PyTorch. The main difference is that you should inherit the `pixyz.distributions.*` class (**Distribution API**), instead of `torch.nn.Module` .

For example, 
<img src="https://latex.codecogs.com/gif.latex?p_{\theta}(x|z)" />
(Bernoulli) 
and
<img src="https://latex.codecogs.com/gif.latex?q_{\phi}(z|x)" />
(normal) are implemented as follows.

```python
>>> from pixyz.distributions import Bernoulli, Normal
>>> # inference model (encoder) q(z|x)
>>> class Inference(Normal):
...     def __init__(self):
...         super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")  # var: variables of this distribution, cond_var: coditional variables.
...         self.fc1 = nn.Linear(784, 512)
...         self.fc21 = nn.Linear(512, 64)
...         self.fc22 = nn.Linear(512, 64)
... 
...     def forward(self, x):  # the name of this argument should be same as cond_var.
...         h = F.relu(self.fc1(x))
...         return {"loc": self.fc21(h),
...                 "scale": F.softplus(self.fc22(h))}  # return parameters of the normal distribution
... 
>>> # generative model (decoder) p(x|z)    
>>> class Generator(Bernoulli):
...     def __init__(self):
...         super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
...         self.fc1 = nn.Linear(64, 512)
...         self.fc2 = nn.Linear(512, 128)
... 
...     def forward(self, z):  # the name of this argument should be same as cond_var.
...         h = F.relu(self.fc1(z))
...         return {"probs": F.sigmoid(self.fc2(h))}    # return a parameter of the Bernoulli distribution
```
Once defined, you can create instances of these classes.
```python
>>> p = Generator()
>>> q = Inference()
```

In VAE,
<img src="https://latex.codecogs.com/gif.latex?p(z)" />
, a prior of the generative model,  is usually defined as the standard normal distribution, without using DNNs. 
Such an instance can be created from `pixyz.distributions.*` as
```python
>>> prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
...                var=["z"], features_shape=[64], name="p_prior")
```

If you want to find out what kind of distribution each instance defines and what modules (the network architecture) define it, just `print` them.
```python
>>> print(p)
Distribution:
  p(x|z)
Network architecture:
  Generator(
    name=p, distribution_name=Bernoulli,
    var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
    (fc1): Linear(in_features=64, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=784, bias=True)
  )
```
If you are working on the iPython environment, you can use `print_latex` to display them in the LaTeX compiled format.

![p](https://user-images.githubusercontent.com/11865486/59156055-1c0dae00-8ad0-11e9-9eac-5b9938904a0d.png)

Conveniently, each distribution instance can **perform sampling** over given samples, regardless of the form of the internal DNN modules. 
```python
>>> samples_z = prior.sample(batch_n=1)
>>> print(samples_z)
{'z': tensor([[ 0.6084,  1.4716,  0.6413,  1.3184, -0.8930,  0.0603,  1.2254,  0.5910, ..., 0.8389]])}
>>> samples = p.sample(samples_z)
>>> print(samples)
{'z': tensor([[ 1.5377,  0.4713,  0.0354,  0.5013,  1.2584,  0.8908,  0.6323,  1.0844, ..., -0.7603]]),
 'x': tensor([[0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., ..., 0.]])}
```
As in this example, samples are represented in dictionary forms in which the keys correspond to random variable names and the values are their realized values.

Moreover, the instance of joint distribution
<img src="https://latex.codecogs.com/gif.latex?p_{\theta}(x,z)=p_{\theta}(x|z)p(z)" />
can be created by **the product of distribution instances**. 
```python
>>> p_joint = p * prior
```

This instance can be checked as
```python
>>> print(p_joint)
Distribution:
  p(x,z) = p(x|z)p_{prior}(z)
Network architecture:
  Normal(
    name=p_{prior}, distribution_name=Normal,
    var=['z'], cond_var=[], input_var=[], features_shape=torch.Size([64])
    (loc): torch.Size([1, 64])
    (scale): torch.Size([1, 64])
  )
  Generator(
    name=p, distribution_name=Bernoulli,
    var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
    (fc1): Linear(in_features=64, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=784, bias=True)
  )
```
![p_joint](https://user-images.githubusercontent.com/11865486/59156030-d81aa900-8acf-11e9-8b8a-ef2d944722b2.png)

Also, it can perform sampling in the same way. 
```python
>>> p_joint.sample(batch_n=1)
{'z': tensor([[ 1.5377,  0.4713,  0.0354,  0.5013,  1.2584,  0.8908,  0.6323,  1.0844, ..., -0.7603]]),
 'x': tensor([[0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., ..., 0.]])}
```
By constructing the joint distribution in this way, you can easily implement **more complicated generative models**.

### 2. Set the loss function of a model(Loss API)
Next, we set the objective (loss) function of the model with defined distributions.

**Loss API** (`pixyz.losses.*`) enables you to define such loss function as if just writing mathematic formulas. The loss function of VAE (Eq.(1)) can easily be converted to the code style  as follows.
```python
>>> from pixyz.losses import KullbackLeibler, LogProb, Expectation as E
>>> reconst = -E(q, LogProb(p)) # the reconstruction loss (it can also be written as `-p.log_prob().expectation()` or `StochasticReconstructionLoss(q, p)`)
>>> kl = KullbackLeibler(q, prior) # Kullback-Leibler divergence
>>> loss_cls = (kl + reconst).mean()
```

Like Distribution API, you can check the formula of the loss function by printing.
```python
>>> print(loss_cls)
mean \left(D_{KL} \left[q(z|x)||p_{prior}(z) \right] - \mathbb{E}_{q(z|x)} \left[\log p(x|z) \right] \right) 
```
![loss](https://user-images.githubusercontent.com/11865486/59156066-3f385d80-8ad0-11e9-9604-ee78a5dd7407.png)

When evaluating this loss function given data, use the `eval` method.
```python
>>> loss_tensor = loss_cls.eval({"x": x_tensor}) # x_tensor: input data
>>> print(loss_tensor)
tensor(1.00000e+05 * 1.2587)
```
### 3. Train the model(Model API)
Finally, Model API (`pixyz.models.Model`) can train the loss function given the optimizer, distributions to train, and training data.
```python
>>> from pixyz.models import Model
>>> from torch import optim
>>> model = Model(loss_cls, distributions=[p, q],
...               optimizer=optim.Adam, optimizer_params={"lr":1e-3}) # initialize a model
>>> train_loss = model.train({"x": x_tensor}) # train the model given training data (x_tensor) 
```
After training the model, you can perform generation and inference on the model by sampling from
<img src="https://latex.codecogs.com/gif.latex?p_{\theta}(x,z)" />
and
<img src="https://latex.codecogs.com/gif.latex?q_{\phi}(z|x)" />
, respectively.

## More information
These frameworks of Pixyz allow the implementation of more complex deep generative models.
See [sample codes](https://github.com/masa-su/pixyz/tree/master/examples) and the [pixyzoo](https://github.com/masa-su/pixyzoo) repository as examples.

For more detailed usage, please check the [Pixyz documentation](https://docs.pixyz.io).

If you encounter some problems in using Pixyz, please let us know.

## Acknowledgements
This library is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO). 
