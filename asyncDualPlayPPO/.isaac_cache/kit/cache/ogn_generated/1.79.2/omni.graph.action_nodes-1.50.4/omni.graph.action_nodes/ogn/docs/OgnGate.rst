.. _omni_graph_action_Gate_2:

.. _omni_graph_action_Gate:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Gate
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action gate


Gate
====

.. <description>

This node controls a flow of execution based on the state of its gate. The gate can be opened or closed by activation of the 'Toggle' gate controls. Each time 'Enter' is activated, the node will activate the 'Exit' signal if the gate is open, and silently succeed if the gate is closed. The current state of the gate is not directly accessible.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Enter (*inputs:enter*)", "``execution``", "Signal to the graph that this node is ready to be executed. Before this signal is activated the gate will be in the state specified by the 'Start Closed' value.", "None"
    "Start Closed (*inputs:startClosed*)", "``bool``", "If true the gate will start in a closed state.", "False"
    "Toggle (*inputs:toggle*)", "``execution``", "Signal to the node that the state of the gate should switch from open to closed, or vice versa. This signal will not activate the 'Exit', it will only determine whether or not the next 'Enter' will activate it.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exit (*outputs:exit*)", "``execution``", "When 'Enter' is activated and the gate is open signal to the graph that execution should continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Gate"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Gate"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnGateDatabase"
    "Python Module", "omni.graph.action_nodes"

