Quickstart
==========

Requirement
-----------

This project comprises a set of commands to be run at a shell command prompt.
Examples used here are for a bash shell in an Ubuntu GNU/Linux environment.

- `Python 3.9 <https://www.python.org/downloads/release/python-390/>`_, see the full list of dependencies in `environment.yml <https://github.com/NRCan/geo-deep-learning/tree/develop/environment.yml>`_
- `hydra <https://hydra.cc/docs/intro/>`_
- `mlflow <https://mlflow.org/>`_
- `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (highly recommended)
- nvidia GPU (highly recommended)

.. note::
   
   The system can be used on your workstation or cluster.

.. _installation:

Installation
------------
Miniconda is suggested as the package manager for GDL. However, users are advised to `switch to libmamba <https://github.com/NRCan/geo-deep-learning#quickstart-with-conda>`_ as conda's default solver or to directly use mamba instead of conda if they are facing extended installation time or other issues. Additional problems are grouped in the `troubleshooting section <https://github.com/NRCan/geo-deep-learning#troubleshooting>`_. If issues persist, users are encouraged to open a new issue for assistance.

Quickstart with conda

To execute scripts in this project, first create and activate your 
python environment with the following commands:

.. code-block:: console

   $ conda env create -f environment.yml
   $ conda activate geo_deep_env

.. note::

   Tested on Ubuntu 20.04, Windows 10 and WSL 2.

Change conda's default solver for faster install (Optional)

.. code-block:: console

   $ conda install -n base conda-libmamba-solver
   $ conda config --set solver libmamba

.. _troubleshooting:
 
 Troubleshooting
----------------

 .. code-block:: console
   $ *ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found*

.. code-block:: console
   $ # Export path to library or set it permenantly in your .bashrc file (example with conda) :
   $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


.. _runninggdl:

Running GDL
-----------

This is an example of how to run **GDL** for a :ref:`segmentation` task on 
the *massachusetts buildings* dataset (`link <https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset>`_).  
**GDL** is using `Hydra library <https://hydra.cc/>`_ for more information 
see the :ref:`configuration` section or go visit their documentation.

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
``./config/gdl_config_template.yaml``. 
For further examples on configuration options or how to change the configuration 
go see the :ref:`configuration` documentation.
