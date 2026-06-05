.. _omni_graph_action_OnClosing_2:

.. _omni_graph_action_OnClosing:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Closing
    :keywords: lang-en omnigraph node graph:action,event threadsafe compute-on-request action on-closing


On Closing
==========

.. <description>

Activates an output signal when the USD stage is about to be closed.
Note that only simple necessary actions should be taken during closing since the application is in the process of cleaning up the existing state and some systems may be in a transitional state.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Closing (*outputs:execOut*)", "``execution``", "After the file close event was received signal to the graph that execution should continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnClosing"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Closing"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnClosingDatabase"
    "Python Module", "omni.graph.action_nodes"

