.. _omni_graph_action_Branch_2:

.. _omni_graph_action_Branch:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Branch
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action branch


Branch
======

.. <description>

Activates an execution output signal along a branch based on a boolean condition.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Condition (*inputs:condition*)", "``bool``", "The boolean condition which determines the output direction.", "False"
    "Input execution (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "False (*outputs:execFalse*)", "``execution``", "When 'Condition' is False signal to the graph that execution can continue downstream.", "None"
    "True (*outputs:execTrue*)", "``execution``", "When 'Condition' is True signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Branch"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Branch"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnBranchDatabase"
    "Python Module", "omni.graph.action_nodes"

