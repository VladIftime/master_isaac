.. _omni_graph_action_Sequence_2:

.. _omni_graph_action_Sequence:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Sequence
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action sequence


Sequence
========

.. <description>

Activates one of two downstream graphs, alternating between the two. No consideration is made whether there is actually a graph downstream of either output.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "A (*outputs:a*)", "``execution``", "On odd-numbered executions, signal to the graph that execution can continue downstream.", "None"
    "B (*outputs:b*)", "``execution``", "On even-numbered executions, signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Sequence"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "hidden", "true"
    "uiName", "Sequence"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnSequenceDatabase"
    "Python Module", "omni.graph.action_nodes"

