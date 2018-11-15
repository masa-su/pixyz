.. role:: hidden
    :class: hidden-section

pixyz.distributions (Distribution API)
======================================

.. automodule:: pixyz.distributions
.. currentmodule:: pixyz.distributions

Distribution
---------------------

.. autoclass:: Distribution
    :members:
    :undoc-members:

Exponential families
---------------------


Normal
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: Normal
    :members:
    :undoc-members:

Bernoulli
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: Bernoulli
    :members:
    :undoc-members:

RelaxedBernoulli
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: RelaxedBernoulli
    :members:
    :undoc-members:

FactorizedBernoulli
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: FactorizedBernoulli
    :members: forward, input_var, get_params, sample, log_likelihood
    :show-inheritance:

Categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: Categorical
    :members: forward, input_var, get_params, sample, log_likelihood
    :show-inheritance:

RelaxedCategorical
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: RelaxedCategorical
    :members: forward, input_var, get_params, sample, log_likelihood
    :show-inheritance:

Special distributions
---------------------

Deterministic
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Deterministic
    :members: forward, input_var, sample
    :show-inheritance:

DataDistribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DataDistribution
    :members: input_var, sample
    :show-inheritance:

NormalPoE
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormalPoE
    :show-inheritance:

CustomLikelihoodDistribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CustomLikelihoodDistribution
    :members: input_var, log_likelihood
    :show-inheritance:

Flow-based
------------------------


PlanarFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: PlanarFlow
    :members: forward, input_var, sample, log_likelihood
    :show-inheritance:


RealNVP
~~~~~~~~~~~~~~~~~~~~~~~~~~
       
.. autoclass:: RealNVP
    :members: forward, input_var, sample, sample_inv, log_likelihood
    :show-inheritance:

Distributions receiving distributions
-------------------------------------


Functions
------------------------

.. currentmodule:: pixyz.distributions.distributions
.. autofunction:: sum_samples

