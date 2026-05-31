.. _omni_graph_action_OnLoaded_2:

.. _omni_graph_action_OnLoaded:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Loaded
    :keywords: lang-en omnigraph node graph:action,event threadsafe compute-on-request action on-loaded


On Loaded
=========

.. <description>

Activates the output on the first update of the graph after it is created or loaded. This will run before any other event node, and will only run once after this node is created.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "On first create or load of the graph signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnLoaded"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Loaded"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnLoadedDatabase"
    "Python Module", "omni.graph.action_nodes"

