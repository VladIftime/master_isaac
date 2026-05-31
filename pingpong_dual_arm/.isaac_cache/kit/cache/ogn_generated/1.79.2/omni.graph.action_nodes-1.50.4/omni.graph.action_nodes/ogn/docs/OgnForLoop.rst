.. _omni_graph_action_ForLoop_2:

.. _omni_graph_action_ForLoop:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: For Loop
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action for-loop


For Loop
========

.. <description>

Executes the a loop body once for each value within a range. When step is positive, the values in the range are determined by the formula:
r[i] = start + step*i, i >= 0 & r[i] < stop.
When step is negative the constraint  is instead r[i] > stop. A step of zero is an error.
The break input can be used to break out of the loop before the last index.  The finished output is executed after all iterations are complete, or when the loop was broken. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Value' and 'Index' outputs.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Break (*inputs:breakLoop*)", "``execution``", "Signal to the graph that execution of the loop body is to be aborted. It behaves exactly as the 'break' statements in Python or C++ behave. After the loop is broken the current 'Value' and 'Index' retain their current values and the 'Finished' signal is activated.", "None"
    "In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"
    "Start (*inputs:start*)", "``int``", "The first value in the range to loop over.", "0"
    "Step (*inputs:step*)", "``int``", "The step size of the range. This number is added to the 'Value' after each execution. The value can be negative to step backwards, however it is an error if the value is zero.", "1"
    "Stop (*inputs:stop*)", "``int``", "The limiting value of the range to loop over. It may or may not be equal to the final value of the loop, depending on the 'Step' value. For example if 'Start' is 1, 'Step' is 2, and 'Stop' is 4 then the loop will run with output 'Value's equal to 1 and 3, but not 5.", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Finished (*outputs:finished*)", "``execution``", "When either the 'Value' has reached or exceeded 'Stop', or 'Signal Break' has activated signal the graph that execution can continue downstream.", "None"
    "Index (*outputs:index*)", "``int``", "The current value of the index when the loop is active, or the value the index had when the 'Stop' threshold was met or exceeded after the loop has completed. The 'Index' value starts at zero after the first execution and increments by one each time the loop body runs.", "None"
    "Loop Body (*outputs:loopBody*)", "``execution``", "For each execution where the 'Value' is still in range signal the graph that execution can continue downstream.", "None"
    "Value (*outputs:value*)", "``int``", "The current value of the range when the loop is active, or the value that met or exceeded the 'Stop' threshold after the loop has completed.", "None"


State
-----
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "I (*state:i*)", "``int``", "The next index in the range, or -1 when loop is not active.", "-1"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.ForLoop"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Icon", "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForLoop.svg"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "tags", "range"
    "uiName", "For Loop"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnForLoopDatabase"
    "Python Module", "omni.graph.action_nodes"

