# Pixyz: a library for developing deep generative models

![logo](https://user-images.githubusercontent.com/11865486/47983581-31c08c80-e117-11e8-8d9d-1efbd920718c.png)


[![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/masa-su/pixyz)

## What is Pixyz?
Pixyz is developed to implement various kind of deep generative models, which is based on [PyTorch](https://pytorch.org/).

Recently, many papers on deep generation models have been published. However, it is likely to be difficult to reproduce them with codes as there is a gap between mathematical formulas presented in these papers and actual implementation of them. Our objective is to create a new library which enables us to fill this gap and easy to implement these models. With Pixyz, you can implement even more complicated models **just as if writing these formulas**, as shown below.

Our library supports following typical deep generative models.

* Explicit models (likelihood-based)
  * variational autoencoders (variational inference)
  * flow-based models
* Implicit models
  * generative adversarial networks

## Installation
```
$ git clone https://github.com/masa-su/pixyz.git
$ pip install -e pixyz --process-dependency-links
```

## Quick Start

So now, let's create a deep generative model with Pixyz! Here, we consider to implement a variational auto-encoder (VAE) which is one of the most well-known deep generative models. VAE is composed of a inference model (q(z|x)) and a generative model (p(x,z)=p(x|z)p(z)), which are defined by DNNs, and this objective function is as follows.

<img src="https://latex.codecogs.com/gif.latex?E_{q_\phi(z|x)}[\log&space;\frac{p_\theta(x,z)}{q_\phi(z|x)}]&space;\leq&space;\log&space;p(x)" />

In Pixyz, you first need to define the distributions of the model by **Distribution API**.

### Define the distributions
In VAE, you should define the three distributions, q(z|x), p(x|z) and p(z), by DNNs. We can accomplish them like PyTorch by inheriting `pixyz.Distribution` class which itself inherits `torch.nn.Module`.


### Set objective function and train the model.
There are three ways to implement and train/test this model.

1. Model API
2. Loss API
3. using only Distribution API

Upper one is for beginners and lower is for developers/researchers. But whatever you choose, you first need to define the distributions of the model by **Distribution API**.


