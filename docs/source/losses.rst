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

Negative expected value of log-likelihood (entropy)
---------------------------------------------------

CrossEntropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pixyz.losses
.. autoclass:: CrossEntropy
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

LossExpectation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LossExpectation
    :members:
    :undoc-members:

Negative log-likelihood
---------------------------------------------------
       
NLL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NLL
    :members:
    :undoc-members:

Lower bound
----------------------------
    
ELBO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ELBO
    :members:
    :undoc-members:

Divergence
----------------------------

KullbackLeibler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KullbackLeibler
    :members:
    :undoc-members:


Similarity 
----------------------------
    
SimilarityLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SimilarityLoss
    :members:
    :undoc-members:

MultiModalContrastivenessLoss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiModalContrastivenessLoss
    :members:
    :undoc-members:

Adversarial loss (GAN loss)
----------------------------

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
