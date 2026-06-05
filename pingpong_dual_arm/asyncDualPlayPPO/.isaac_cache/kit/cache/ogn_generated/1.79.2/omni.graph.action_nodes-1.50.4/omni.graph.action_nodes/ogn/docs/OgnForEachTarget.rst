.. _omni_graph_action_ForEachTarget_1:

.. _omni_graph_action_ForEachTarget:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: For Each Target Loop
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action for-each-target


For Each Target Loop
====================

.. <description>

Activates the 'Loop Body' signal once for each target in 'Targets', making the current array member available in the output 'Target' with its index in 'Array Index'. After every element of 'Targets' has been processed the 'Finished' signal is activated. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Target' output.

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
    "Targets (*inputs:targets*)", "``target``", "The targets array to loop over", "None"
    "", "Metadata", "*allowMultiInputs* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Array Index (*outputs:arrayIndex*)", "``int``", "The value of the current index being visited by the loop. Keeps the value of the last index after the loop has completed. The index starts at zero and increments by one as it walks through the members of 'Targets'.", "None"
    "Finished (*outputs:finished*)", "``execution``", "When the final element of 'Targets' has been processed signal the graph that execution can continue downstream.", "None"
    "Loop Body (*outputs:loopBody*)", "``execution``", "For each member of 'Targets' signal the graph that execution can continue downstream.", "None"
    "Target (*outputs:target*)", "``target``", "The current member of 'Targets' being visited by the loop. Keeps the value of the last array element after the loop has completed.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.ForEachTarget"
    "Version", "1"
    "Extension", "omni.graph.action_nodes"
    "Icon", "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForEachTarget.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "For Each Target Loop"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnForEachTargetDatabase"
    "Python Module", "omni.graph.action_nodes"

