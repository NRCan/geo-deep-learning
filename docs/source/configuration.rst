.. _configuration:

Configuration
+++++++++++++

How to use **Hydra** config files
---------------------------------

Since **GDL** use the `Hydra <https://hydra.cc/docs/intro/#quick-start-guide>`_ library, here's how
**GDL** configuration work. Through a main ``yaml`` file 
`gdl_config_template.yaml <https://github.com/NRCan/geo-deep-learning/tree/develop/config/gdl_config_template.yaml>`_ 
located in the `config <https://github.com/NRCan/geo-deep-learning/tree/develop/config/>`_ 
folder which handles additional ``yaml`` files located in other subfolders.
We recommend starting with the 
`gdl_config_template.yaml <https://github.com/NRCan/geo-deep-learning/tree/develop/config/gdl_config_template.yaml>`_ 
as the configuration entrypoint. To run your code with the proper parameters, see the 
:ref:`Running GDL <runninggdl>` section for an example with a semantic segmentation task. 
There are other examples at the end of this document in the :ref:`confexample` section.

The ``config`` folder is structured as depicted below. 
It is important to remember that the 
`gdl_config_template.yaml <https://github.com/NRCan/geo-deep-learning/tree/develop/config/gdl_config_template.yaml>`_  
file contains every parameter necessary for executing the command. 
Other ``yaml`` files in subfolders handle specific categories of parameters. 

| config
| ├── gdl_config_template.yaml
| ├── model
|     ├── gdl_unet.yaml
|     └── smp_deeplabv3.yaml
|     ...
| └── task
|     └── segmentation.yaml
| 

The code is currently executed with 
`gdl_config_template.yaml <https://github.com/NRCan/geo-deep-learning/tree/develop/config/gdl_config_template.yaml>`_ 
as a default configuration, the :ref:`confexample` section have an example on how to run the script with 
an other config ``yaml``. 

Each configuration file need to follow the structure show bellow, and each of those sections
have their functionality that will be explain.

.. code-block:: yaml

    defaults:
        ...
    general:
        ...
    inference:
        ...
    print_config: ...
    mode:  ...
    debug: ...

.. _configurationdefaultparam:

Defaults Parameters
===================

The *defaults* section is where all default ``yaml`` files are loaded as input values for each category of parameters, 
you don't have to specify all of them.
For example, ``model: gdl_unet`` means the ``config`` folder contains a subfolder called ``model`` which have a ``gdl_unet.yaml`` file.
So for every time that **GDL** code call the parameter ``model``, it will have all the variables set in the ``gdl_unet.yaml``.
If you want to run **GDL** with another model like ``smp_deeplabv3``, you only have to change ``model: gdl_unet`` to ``model: smp_deeplabv3``.
Just be sure that you have ``smp_deeplabv3.yaml`` in your ``model`` folder 
(the :ref:`confexample` section have an example on how to change this parameter in the command line).
Options for each category of parameters are found in config subfolder by the same name.

.. code-block:: yaml

    defaults:
        - model: gdl_unet
        - verify: default_verify
        - tiling: default_tiling
        - training: default_training
        - loss: binary/softbce
        - optimizer: adamw
        - callbacks: default_callbacks
        - scheduler: plateau
        - dataset: test_ci_segmentation_binary
        - augmentation: basic_augmentation_segmentation
        - tracker: # set logger here or use command line (e.g. `python GDL.py tracker=mlflow`)
        - visualization: default_visualization
        - inference: default_binary
        - hydra: default
        - override hydra/hydra_logging: colorlog # enable color logging to make it pretty
        - override hydra/job_logging: colorlog # enable color logging to make it pretty
        - _self_

All the files in the *defaults* section can be overwritten on the command line,
go to :ref:`confexample` section too see how to do.
The main goal of the structure is to organize all the parameters in meaningful and logical categories.
If you want to add new options for a category, 
you'll need to include ``# @package _global_`` at the beginning of each ``yaml`` added. 
By doing so, the code in python will read ``model.parameters_name`` as a directory.
If you accidentally omit the prefix ``# @package _global_``, 
the python code will read ``model.unet.parameters_name`` (as set by default currently), 
so to be more versatile we want to read ``model.parameters_name``.
For example if you created ``new_model.yaml`` to be read as a model and you don't want
to change the main code to read this file each time you change model.
For more information about packages in Hydra,
see `Hydra's documentation on Packages <https://hydra.cc/docs/advanced/overriding_packages>`_.

For the ``tiling`` parameter, you can find more information in the :ref:`datatiling`
containing the information to execute the this job.
Same for the ``training`` and ``inference`` parameter, the information can be found at 
the :ref:`training` and :ref:`inference` section respectively.
When *training* the ``inference`` part doesn't need to be filled 
and vice versa.

The ``tracker`` is set to nothing by default, but will still log the information in the log folder.
If you want to set a tracker you can change the value in the config file or add the tracker
parameter at execution time via the command line ``python GDL.py tracker=mlflow mode=train``.
We recommend to use ``mlflow``, since the development team use it, but you can use whatever you want and 
create a ``yaml`` for it.

.. _configurationgeneralparam:

General Parameters
==================

This section contains general parameters information that will be read by the code,
normally contain parameters often changed or paths to important file.  
Other ``yaml`` files from the *defaults* section will read parameters from the *general* section.

.. code-block:: yaml

    task: segmentation
    work_dir: ${hydra:runtime.cwd}  # where the code is executed
    config_name: ${hydra:job.config_name}
    config_override_dirname: ${hydra:job.override_dirname}
    config_path: ${hydra:runtime.config_sources}
    project_name: template_project
    workspace: your_name
    max_epochs: 2 # for train only
    min_epochs: 1 # for train only
    raw_data_dir: data
    raw_data_csv: tests/tiling/tiling_segmentation_binary_ci.csv
    tiling_data_dir: ${general.raw_data_dir}/patches # where the patches will be saved
    save_weights_dir: saved_model/${general.project_name}

.. note::
    The ``task`` parameter have multiple options, see the :ref:`taskindex`  section.


Print Config Parameter
======================

If ``True``, this will save the config inder the run subfolder generated in the log folder.

Mode Parameter
==============

.. code-block:: yaml

    mode: {verify, tiling, train, inference, evaluate}

For **GDL**, the modes available are:
    - *verify*, verify the given data and generate an ``csv`` with infos and stats on those images.
    - *tiling*, generates tiles from each source aoi (image & ground truth).
    - *train*, will train the model specified with all the parameters in the configuration file.
    - *inference*, generate the inference for the given images.
    - *evaluate*, generate statistics on the given images, unlike the *inference*, this mode need the images to be link to a ground truth.
    
.. note::
    Each of those modes will be different for all the tasks, for further information on the well 
    being of those modes, see the :ref:`modeindex`  section.

Debug Parameter
===============

If ``True``, this will print the complete yaml config at the beginning 
plus run a validation test on the dataloader before the training.

.. _confexample:

Examples
--------

Here some examples on how to run GDL with Hydra.

Basic usage
===========

Run the code with all the defaults value in the
`gdl_config_template.yaml <https://github.com/NRCan/geo-deep-learning/tree/develop/config/gdl_config_template.yaml>`_ .

.. code-block:: console

   (geo_deep_env) $ python GDL.py mode=train


Overwritting only one parameter
===============================

Changing only one parameter in the configuration.

.. code-block:: console

    # Changing the number of max epochs for training
    (geo_deep_env) $ python GDL.py mode=train general.max_epochs=100
    # Changing the dropout for the chosen model
    (geo_deep_env) $ python GDL.py mode=train model.dropout=True

Adding a new parameters
=======================

Adding a new parameters in the config without having to write it in the ``yaml``. 

.. code-block:: console

    (geo_deep_env) $ python GDL.py mode=train +new.params=1.0

The configuration that will be save for this run will look like that: 

.. code-block:: yaml

    defaults:
        ...

    general:
        ...
        
    print_config:  ...
    mode:  ...
    debug: ...
    new:
        params: 1.0

Using an other configuration file
=================================

How to using a new ``gdl_config.yaml`` file that has the same structure as the template ``yaml`` 
but have different values. The usecase for that is, for example, you have a certain configuration
for your pipline that is different form your testing one, you dont want to change your parameters 
each time. So you create a new ``yaml`` for your pipline and when you are ready to run it, you 
only have to run it like that: 

.. code-block:: console

   (geo_deep_env) $ python GDL.py --config-name=/path/to/new/gdl_pipline_config.yaml mode=train


Other Hydra parameters to overwrite
===================================

See `Hydra's documentation on command line flags <https://hydra.cc/docs/advanced/hydra-command-line-flags/>`_
page for more informations.

