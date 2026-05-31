.. _omni_graph_action_SetPrimActive_2:

.. _omni_graph_action_SetPrimActive:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Set Prim Active
    :keywords: lang-en omnigraph node graph:action,sceneGraph WriteOnly action set-prim-active


Set Prim Active
===============

.. <description>

Set whether a prim on the Stage is active (selected) or not. Only one prim can be connected to the 'Prim' input for execution. If multiple targets are present then a warning will be logged. The 'Active' input value will be the prim's new active state. If the prim cannot be found then an error will be logged. Avoiding use of the deprecated 'Prim Path' input will ensure that error cannot be encountered.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Active (*inputs:active*)", "``bool``", "Whether to set the prim active or not", "False"
    "Exec In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"
    "Prim Path (*inputs:prim*)", "``path``", "The prim to be (de)activated", ""
    "Prim (*inputs:primTarget*)", "``target``", "The prim to be (de)activated", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Execute Out (*outputs:execOut*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.SetPrimActive"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Set Prim Active"
    "Categories", "graph:action,sceneGraph"
    "Generated Class Name", "OgnSetPrimActiveDatabase"
    "Python Module", "omni.graph.action_nodes"

