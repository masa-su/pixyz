pixyz.losses (Loss API)
=======================

.. automodule:: pixyz.losses
.. currentmodule:: pixyz.losses

Loss
----------------------------

.. currentmodule:: pixyz.losses.losses
.. autoclass:: Loss
    :members:
    :undoc-members:

Probability density function
----------------------------

LogProb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pixyz.losses
.. autoclass:: LogProb
    :members:
    :undoc-members:

Prob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Prob
    :members:
    :undoc-members:

Expected value
--------------

Expectation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Expectation
    :members:
    :undoc-members:

REINFORCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: REINFORCE

Entropy
--------

Entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: Entropy

CrossEntropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: CrossEntropy


Lower bound
----------------------------
    
ELBO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ELBO

Statistical distance
----------------------------

KullbackLeibler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: KullbackLeibler

WassersteinDistance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: WassersteinDistance
    :members:
    :undoc-members:

MMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MMD
    :members:
    :undoc-members:


Adversarial statistical distance
--------------------------------

AdversarialJensenShannon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdversarialJensenShannon
    :members:
    :undoc-members:

AdversarialKullbackLeibler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdversarialKullbackLeibler
    :members:
    :undoc-members:       

AdversarialWassersteinDistance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdversarialWassersteinDistance
    :members:
    :undoc-members:


Loss for sequential distributions
-----------------------------------

IterativeLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: IterativeLoss
    :members:
    :undoc-members:

Loss for special purpose
----------------------------

Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pixyz.losses.losses
.. autoclass:: Parameter
    :members:
    :undoc-members:

ValueLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pixyz.losses.losses
.. autoclass:: ValueLoss
    :members:
    :undoc-members:

       
Operators
----------------------------

LossOperator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LossOperator
    :members:
    :undoc-members:

LossSelfOperator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LossSelfOperator
    :members:
    :undoc-members:

AddLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AddLoss
    :members:
    :undoc-members:

SubLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SubLoss
    :members:
    :undoc-members:

MulLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MulLoss
    :members:
    :undoc-members:

DivLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DivLoss
    :members:
    :undoc-members:

MinLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MinLoss
    :members:
    :undoc-members:

MaxLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaxLoss
    :members:
    :undoc-members:

NegLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NegLoss
    :members:
    :undoc-members:

AbsLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AbsLoss
    :members:
    :undoc-members:

BatchMean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchMean
    :members:
    :undoc-members:

BatchSum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchSum
    :members:
    :undoc-members:
