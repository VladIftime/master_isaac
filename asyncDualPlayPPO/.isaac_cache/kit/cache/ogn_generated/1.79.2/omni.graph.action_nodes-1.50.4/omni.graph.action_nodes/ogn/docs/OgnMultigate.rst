.. _omni_graph_action_Multigate_2:

.. _omni_graph_action_Multigate:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Multigate
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action multigate


Multigate
=========

.. <description>

This node cycles through each of its N outputs. On each input, one output will be activated. Outputs will be activated in sequence, eg: 0->1->2->3->4->0->1.... 'Output 0' is provided as the first output to be activated. To add more to the sequence add new output attributes with indexed unique names, such as 'outputs:output1', 'outputs:output2', etc.

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
    "Reset (*inputs:reset*)", "``execution``", "Signal to the node to reset its internal counter. The next time 'Execute In' is activated it will go back to activating 'Output 0'. This will skip any other outputs present.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Output0 (*outputs:output0*)", "``execution``", "On the first execution signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Multigate"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Multigate"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnMultigateDatabase"
    "Python Module", "omni.graph.action_nodes"

