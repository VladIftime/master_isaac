.. _omni_graph_action_SyncGate_2:

.. _omni_graph_action_SyncGate:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Sync Gate
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action sync-gate


Sync Gate
=========

.. <description>

Activate the downstream graphs after all of the input signals have been triggered with the same synchronization value. An internal count is maintained that resets whenever a new synchronization value is encountered. If the count reaches a number equal to the number of 'Execute In' inputs then the 'Execute Out' activation signals to the downstream graph that it is ready to be executed.

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
    "Sync (*inputs:syncValue*)", "``uint64``", "Value against which each of the input signals are checked for synchronization.", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Execute Out (*outputs:execOut*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"
    "Sync (*outputs:syncValue*)", "``uint64``", "Current synchronization value, independent of whether the output activation signal has been triggered.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.SyncGate"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Sync Gate"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnSyncGateDatabase"
    "Python Module", "omni.graph.action_nodes"

