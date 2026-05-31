.. _omni_graph_action_OnImpulseEvent_3:

.. _omni_graph_action_OnImpulseEvent:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Impulse Event
    :keywords: lang-en omnigraph node graph:action,event threadsafe compute-on-request action on-impulse-event


On Impulse Event
================

.. <description>

Triggers the output execution once when the state attribute 'Enable Impulse' is set.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Trigger (*outputs:execOut*)", "``execution``", "After the impulse, signal to the graph that execution can continue downstream.", "None"


State
-----
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Enable Impulse (*state:enableImpulse*)", "``bool``", "When true, activate 'Trigger' once and reset to false.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnImpulseEvent"
    "Version", "3"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Impulse Event"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnImpulseEventDatabase"
    "Python Module", "omni.graph.action_nodes"

