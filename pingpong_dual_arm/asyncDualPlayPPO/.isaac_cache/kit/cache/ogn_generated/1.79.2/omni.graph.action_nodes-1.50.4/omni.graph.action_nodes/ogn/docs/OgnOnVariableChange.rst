.. _omni_graph_action_OnVariableChange_2:

.. _omni_graph_action_OnVariableChange:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Variable Change
    :keywords: lang-en omnigraph node graph:action,event threadsafe action on-variable-change


On Variable Change
==================

.. <description>

Activates execution of the downstream graph when a graph variable's value changes. The name of the variable comes from the input 'Variable Name'. Only variables in the current graph are considered.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""
    "Variable Name (*inputs:variableName*)", "``token``", "The name of the graph variable to monitor for changes.", ""
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Changed (*outputs:changed*)", "``execution``", "When the variable value changes, signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnVariableChange"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Variable Change"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnVariableChangeDatabase"
    "Python Module", "omni.graph.action_nodes"

