.. _omni_graph_action_OnTick_2:

.. _omni_graph_action_OnTick:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Tick
    :keywords: lang-en omnigraph node graph:action,event threadsafe action on-tick


On Tick
=======

.. <description>

Activates execution of the downstream graph at a regular multiple of the application's refresh rate. As the application runs the this node will skip every 'Update Period' executions. Typically you might set this to a larger value if the downstream graph is doing something you do not want to happen too frequently, such as printing out debugging messages or polling some input that does not get triggered very often. The actual refresh rate timing will depend on external factors such as rendering time and scene complexity. In addition to the activation signal, the outputs also contain global time information for convenience.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Update Period (Ticks) (*inputs:framePeriod*)", "``uint``", "The number of application refreshes to skip between executions of this node. For example, the default 0 means no skipping so this node executes on every application update, 1 means every other update. This can be used to rate-limit updates.", "0"
    "", "Metadata", "*literalOnly* = 1", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Absolute Simulation Time (Seconds) (*outputs:absoluteSimTime*)", "``double``", "The accumulated total of elapsed times between rendered frames. This is independent of any executions skipped as a result of a non-zero 'Update Period'.", "None"
    "Delta (Seconds) (*outputs:deltaSeconds*)", "``double``", "The accumulated graph evaluation time since the last time the graph downstream of 'Tick' was activated.", "None"
    "Animation Time (Frames) (*outputs:frame*)", "``double``", "The global animation time in frames, equivalent to (Animation Time * FramesPerSecond), during playback.", "None"
    "Is Playing (*outputs:isPlaying*)", "``bool``", "True during global animation timeline playback.", "None"
    "Tick (*outputs:tick*)", "``execution``", "At each specified update tick, signal to the graph that execution can continue downstream.", "None"
    "Animation Time (Seconds) (*outputs:time*)", "``double``", "The global animation time in seconds during playback.", "None"
    "Time Since Start (Seconds) (*outputs:timeSinceStart*)", "``double``", "Elapsed time since the application started.", "None"


State
-----
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Accumulated Seconds (*state:accumulatedSeconds*)", "``double``", "Accumulated time since the last activation of the downstream graph.", "0"
    "Frame Count (*state:frameCount*)", "``uint``", "Accumulated frames since the last activation of the downstream graph.", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnTick"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Tick"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnTickDatabase"
    "Python Module", "omni.graph.action_nodes"

