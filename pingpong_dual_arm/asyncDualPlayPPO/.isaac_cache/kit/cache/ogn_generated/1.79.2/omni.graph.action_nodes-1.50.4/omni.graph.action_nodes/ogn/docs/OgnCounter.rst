.. _omni_graph_action_Counter_2:

.. _omni_graph_action_Counter:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Counter
    :keywords: lang-en omnigraph node graph:action,function threadsafe action counter


Counter
=======

.. <description>

This node counts the number of times it has been executed since the 'Reset' signal was activated.

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
    "Reset (*inputs:reset*)", "``execution``", "Signal to the node to reset its internal counter.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Count (*outputs:count*)", "``int``", "The number of times this node has been executed since being reset.", "None"
    "Execute Out (*outputs:execOut*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"


State
-----
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Count (*state:count*)", "``int``", "Internal value storing the execution count in a persistent way.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.Counter"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "Counter"
    "Categories", "graph:action,function"
    "Generated Class Name", "OgnCounterDatabase"
    "Python Module", "omni.graph.action_nodes"

