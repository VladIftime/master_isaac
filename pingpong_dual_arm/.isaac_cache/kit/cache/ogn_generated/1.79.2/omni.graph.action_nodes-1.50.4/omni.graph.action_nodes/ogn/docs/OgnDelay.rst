.. _omni_graph_action_Delay_2:

.. _omni_graph_action_Delay:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Delay
    :keywords: lang-en omnigraph node graph:action,time threadsafe action delay


Delay
=====

.. <description>

This node will stop downstream execution for a period of time before continuing. The period of time is in 'Duration', measured in seconds, and begins when 'Execute In' is activated. Once the delay period has elapsed the node activates its 'Finished' signal, and execution continues downstream.
It is important to note that the execution can only resume when the application next updates. As the length of the application update varies based on the complexity, the 'Duration' is merely a minimum delay time. For example if the 'Duration' is 10 seconds and the application update time is 7 seconds then the node will trigger somewhere around the 14 second mark. After the first execution only 7 seconds will have elapsed, less than the 10 seconds requested, and at the second execution it will be a total of 14 seconds, which will be recognized as satisfying the 'Duration' requirement.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Duration (*inputs:duration*)", "``double``", "The duration of the delay in seconds.", "0.0"
    "Execute In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Finished (*outputs:finished*)", "``execution``", "After 'Duration' has elapsed signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Delay"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "Delay"
    "Categories", "graph:action,time"
    "Generated Class Name", "OgnDelayDatabase"
    "Python Module", "omni.graph.action_nodes"

