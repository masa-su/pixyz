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

So now, let's create a deep generative model with Pixyz! Here, we consider to implement a variational auto-encoder (VAE) which is one of the most well-known deep generative models. VAE is composed of a inference model ($p(z|x)$) and a generative model ($p(x,z)=p(x|z)p(z)$), and this objective function is as follows.
