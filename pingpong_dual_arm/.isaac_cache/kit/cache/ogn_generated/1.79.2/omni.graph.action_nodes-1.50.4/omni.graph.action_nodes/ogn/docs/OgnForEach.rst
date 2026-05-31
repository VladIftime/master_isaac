.. _omni_graph_action_ForEach_2:

.. _omni_graph_action_ForEach:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: For Each Loop
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action for-each


For Each Loop
=============

.. <description>

Activates the 'Loop Body' signal once for each element in the 'Input Array', making the current array member available in 'Element' with its index in 'Array Index'. After every element of the 'Input Array' has been processed the 'Finished' signal is activated. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Element' output.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Input Array (*inputs:arrayIn*)", "``['bool[]', 'colord[3][]', 'colord[4][]', 'colorf[3][]', 'colorf[4][]', 'colorh[3][]', 'colorh[4][]', 'double[2][]', 'double[3][]', 'double[4][]', 'double[]', 'float[2][]', 'float[3][]', 'float[4][]', 'float[]', 'frame[4][]', 'half[2][]', 'half[3][]', 'half[4][]', 'half[]', 'int64[]', 'int[2][]', 'int[3][]', 'int[4][]', 'int[]', 'matrixd[2][]', 'matrixd[3][]', 'matrixd[4][]', 'normald[3][]', 'normalf[3][]', 'normalh[3][]', 'pointd[3][]', 'pointf[3][]', 'pointh[3][]', 'quatd[4][]', 'quatf[4][]', 'quath[4][]', 'texcoordd[2][]', 'texcoordd[3][]', 'texcoordf[2][]', 'texcoordf[3][]', 'texcoordh[2][]', 'texcoordh[3][]', 'timecode[]', 'token[]', 'transform[4][]', 'uchar[]', 'uint64[]', 'uint[]', 'vectord[3][]', 'vectorf[3][]', 'vectorh[3][]']``", "The array to loop over", "None"
    "Exec In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Array Index (*outputs:arrayIndex*)", "``int``", "The value of the current index being visited by the loop. Keeps the value of the last index after the loop has completed. The index starts at zero and increments by one as it walks through the members of 'Input Array'.", "None"
    "Element (*outputs:element*)", "``['bool', 'colord[3]', 'colord[4]', 'colorf[3]', 'colorf[4]', 'colorh[3]', 'colorh[4]', 'double', 'double[2]', 'double[3]', 'double[4]', 'float', 'float[2]', 'float[3]', 'float[4]', 'frame[4]', 'half', 'half[2]', 'half[3]', 'half[4]', 'int', 'int64', 'int[2]', 'int[3]', 'int[4]', 'matrixd[2]', 'matrixd[3]', 'matrixd[4]', 'normald[3]', 'normalf[3]', 'normalh[3]', 'pointd[3]', 'pointf[3]', 'pointh[3]', 'quatd[4]', 'quatf[4]', 'quath[4]', 'texcoordd[2]', 'texcoordd[3]', 'texcoordf[2]', 'texcoordf[3]', 'texcoordh[2]', 'texcoordh[3]', 'timecode', 'token', 'transform[4]', 'uchar', 'uint', 'uint64', 'vectord[3]', 'vectorf[3]', 'vectorh[3]']``", "The current member of 'Input Array' being visited by the loop. Keeps the value of the last array element after the loop has completed.", "None"
    "Finished (*outputs:finished*)", "``execution``", "When the final element of 'Input Array' has been processed signal the graph that execution can continue downstream.", "None"
    "Loop Body (*outputs:loopBody*)", "``execution``", "For each member of 'Input Array' signal the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.ForEach"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Icon", "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForEach.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "For Each Loop"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnForEachDatabase"
    "Python Module", "omni.graph.action_nodes"

