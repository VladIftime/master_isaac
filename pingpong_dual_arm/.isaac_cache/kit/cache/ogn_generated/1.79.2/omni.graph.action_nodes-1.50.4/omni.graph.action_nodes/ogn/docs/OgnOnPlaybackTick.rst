.. _omni_graph_action_OnPlaybackTick_2:

.. _omni_graph_action_OnPlaybackTick:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Playback Tick
    :keywords: lang-en omnigraph node graph:action,event threadsafe action on-playback-tick


On Playback Tick
================

.. <description>

For each frame tick during playback, activate the downstream graph execution. In addition to the activation signal, the outputs also contain the playback time values, taken directly from the execution context.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Delta Seconds (*outputs:deltaSeconds*)", "``double``", "The number of seconds that have elapsed since the last update.", "None"
    "Frame (*outputs:frame*)", "``double``", "The global playback time in frames, equivalent to (Time * PlaybackFramesPerSecond).", "None"
    "Tick (*outputs:tick*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"
    "Time (*outputs:time*)", "``double``", "The global playback time in seconds.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnPlaybackTick"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Playback Tick"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnPlaybackTickDatabase"
    "Python Module", "omni.graph.action_nodes"

