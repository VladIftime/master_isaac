.. _omni_graph_action_Multisequence_2:

.. _omni_graph_action_Multisequence:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Sequence
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action multisequence


Sequence
========

.. <description>

Outputs an execution pulse along each of its N outputs in sequence. For every single input execution pulse, each and every output will be exclusively enabled in order. 'Output 0' is provided by default. To add more to the sequence add new output attributes with indexed unique names, such as 'outputs:output1', 'outputs:output2', etc.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Execute In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Output0 (*outputs:output0*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Multisequence"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Sequence"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnMultisequenceDatabase"
    "Python Module", "omni.graph.action_nodes"

