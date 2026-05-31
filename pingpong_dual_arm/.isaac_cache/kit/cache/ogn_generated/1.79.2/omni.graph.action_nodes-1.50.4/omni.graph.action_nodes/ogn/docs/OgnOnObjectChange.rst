.. _omni_graph_action_OnObjectChange_5:

.. _omni_graph_action_OnObjectChange:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On USD Object Change
    :keywords: lang-en omnigraph node graph:action,event threadsafe action on-object-change


On USD Object Change
====================

.. <description>

Monitors a specific 'Property Name' on a connected 'Prim' target. When a change in the underlying USD is detected, activates execution of the downstream graph.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Property Name (*inputs:name*)", "``token``", "The name of the property of interest on the USD prim being monitored.", "None"
    "", "Metadata", "*literalOnly* = 1", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""
    "Path (*inputs:path*)", "``path``", "The path of object of interest (property or prim). If the prim input has a target, this is ignored", "None"
    "", "Metadata", "*literalOnly* = 1", ""
    "Prim (*inputs:prim*)", "``target``", "The USD prim being monitored.", "None"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Changed (*outputs:changed*)", "``execution``", "When the watched property changes signal to the graph that execution can continue downstream.", "None"
    "Property Name (*outputs:propertyName*)", "``token``", "The name of the property on which the change was detected.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnObjectChange"
    "Version", "5"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On USD Object Change"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnObjectChangeDatabase"
    "Python Module", "omni.graph.action_nodes"

