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

### 1, Define the distributions
In VAE, you should define the three distributions, q(z|x), p(x|z) and p(z), by DNNs. We can accomplish them like PyTorch by inheriting `pixyz.Distribution` class which itself inherits `torch.nn.Module`.

For example, p(x|z) (Bernoulli) and q(z|x) (Normal) can be defined as follows.

```python
from pixyz.distributions import Bernoulli, Normal
# inference model q(z|x)
class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(cond_var=["x"], var=["z"], name="q")        

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, 64)
        self.fc32 = nn.Linear(512, 64)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    
# generative model p(x|z)    
class Generator(Bernoulli):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 128)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": F.sigmoid(self.fc3(h))}
```
Once defined, we can create instances of distributions. 
```python
p = Generator()
q = Inference()
```

If you want to use distributions which don't need to be defined with DNNs (simple distribution), you just create new instance from `pixyz.Distribution`.

```python
loc = torch.tensor(0.)
scale = torch.tensor(1.)
prior = Normal(loc=loc, scale=scale, var=["z"], dim=64, name="p_prior")
```


You can check by printing the type of distribution defined by each instance.
```python
print(p)
>> Distribution:
>>   p(x|z) (Bernoulli)
>> Network architecture:
>>   Generator(
>>     (fc1): Linear(in_features=64, out_features=512, bias=True)
>>     (fc2): Linear(in_features=512, out_features=512, bias=True)
>>     (fc3): Linear(in_features=512, out_features=784, bias=True)
>> )
```

Convinentlly, we can **sample** some variables and **estimate the (log-)likelihood** of each instance, which will be explained in later (see sec. 2.3).


### 2. Set objective function and train the model
There are three ways to implement and train/test this model.

1. Model API
2. Loss API
3. using only Distribution API

Upper one is for beginners and lower is for developers/researchers.

#### 2.1. Model API

#### 2.2. Loss API

#### 2.3. using only Distribution API



