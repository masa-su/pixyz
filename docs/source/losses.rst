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

Entropy
--------

EmpiricalCrossEntropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EmpiricalCrossEntropy
    :members:
    :undoc-members:

EmpiricalEntropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EmpiricalEntropy
    :members:
    :undoc-members:

Entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Entropy
    :members:
    :undoc-members:

StochasticReconstructionLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StochasticReconstructionLoss
    :members:
    :undoc-members:

Lower bound
----------------------------
    
ELBO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ELBO
    :members:
    :undoc-members:

Statistical distance
----------------------------

EmpiricalKullbackLeibler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EmpiricalKullbackLeibler
    :members:
    :undoc-members:

KullbackLeibler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KullbackLeibler
    :members:
    :undoc-members:

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

SetLoss
~~~~~~~~~~~

.. autoclass:: SetLoss
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
