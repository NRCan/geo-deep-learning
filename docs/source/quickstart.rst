Quickstart
==========

Requirement
-----------

This project comprises a set of commands to be run at a shell command prompt.
Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- `Python 3.9 <https://www.python.org/downloads/release/python-390/>`_, see the full list of dependencies in [environment.yml](environment.yml)
- `hydra <https://hydra.cc/docs/intro/>`_
- `mlflow <https://mlflow.org/>`_
- `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (highly recommended)
- nvidia GPU (highly recommended)

.. note::
   
   The system can be used on your workstation or cluster.

.. _installation:

Installation
------------

To execute scripts in this project, first create and activate your 
python environment with the following commands:

.. code-block:: console

   $ conda env create -f environment.yml
   $ conda activate geo_deep_env

.. note::

   Tested on Ubuntu 20.04 and Windows 10 using miniconda.

Running GDL
-----------

This is an example of how to run GDL for a :ref:`segmentation` task on 
the ``massachusetts buildings`` dataset (add the link).  
GDL is using `Hydra <https://hydra.cc/>`_ library for more information 
see their documentation.

.. code-block:: console

   # Clone this github repo
   (geo_deep_env) $ git clone https://github.com/NRCan/geo-deep-learning.git
   (geo_deep_env) $ cd geo-deep-learning

   # By default the task is set to segmentation
   # Creating the patches from the raw data
   (geo_deep_env) $ python GDL.py mode=tiling
   # Training the neural network
   (geo_deep_env) $ python GDL.py mode=train
   # Inference on the data
   (geo_deep_env) $ python GDL.py mode=inference

This example runs with a default configuration
`./config/gdl_config_template.yaml <./config/gdl_config_template.yaml>`_. 
For further examples on configuration options or how to change the configuration 
go see the :ref:`configuration` documentation.

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

