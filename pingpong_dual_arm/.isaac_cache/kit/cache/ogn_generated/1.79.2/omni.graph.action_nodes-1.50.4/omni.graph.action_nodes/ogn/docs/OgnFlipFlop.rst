.. _omni_graph_action_FlipFlop_2:

.. _omni_graph_action_FlipFlop:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Flip Flop
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action flip-flop


Flip Flop
=========

.. <description>

This node activates its outputs in an alternating sequence, starting with 'On Odd' on the first execution after 'Signal Execution' has been activated.

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

    "Execute A (*outputs:a*)", "``execution``", "After every odd-numbered execution signal to the graph that execution can continue downstream.", "None"
    "Execute B (*outputs:b*)", "``execution``", "After every even-numbered execution signal to the graph that execution can continue downstream.", "None"
    "Is A (*outputs:isA*)", "``bool``", "Set to true when the 'Execute A' signal is active, otherwise set to false.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.FlipFlop"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "Flip Flop"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnFlipFlopDatabase"
    "Python Module", "omni.graph.action_nodes"

