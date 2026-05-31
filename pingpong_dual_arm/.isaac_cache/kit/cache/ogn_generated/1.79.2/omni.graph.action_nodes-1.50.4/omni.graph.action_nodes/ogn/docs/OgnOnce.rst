.. _omni_graph_action_Once_2:

.. _omni_graph_action_Once:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Once
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action once


Once
====

.. <description>

Controls flow of execution by activating the 'Once' signal on the first execution and the 'After' signal for each successive execution.

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
    "Reset (*inputs:reset*)", "``execution``", "Signal to the node to reset the state so that 'Once' will be activated on the next 'Exec In'.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "After (*outputs:after*)", "``execution``", "On every execution after the first one, or the first execution after 'Reset' is activated signal to the graph that execution can continue downstream.", "None"
    "Once (*outputs:once*)", "``execution``", "On the very first execution, or the first execution after 'Reset' is activated signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Once"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Once"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnOnceDatabase"
    "Python Module", "omni.graph.action_nodes"

