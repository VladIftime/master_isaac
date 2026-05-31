.. _omni_graph_action_OnGamepadInput_2:

.. _omni_graph_action_OnGamepadInput:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Gamepad Input
    :keywords: lang-en omnigraph node graph:action,input:gamepad compute-on-request action on-gamepad-input


On Gamepad Input
================

.. <description>

Event node which fires when a gamepad event occurs. This node only capture events on buttons, excluding triggers and sticks.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Gamepad Element In (*inputs:gamepadElementIn*)", "``token``", "The gamepad button that will trigger the downstream execution.", "Face Button Bottom"
    "", "Metadata", "*displayGroup* = parameters", ""
    "", "Metadata", "*literalOnly* = 1", ""
    "", "Metadata", "*allowedTokens* = Face Button Bottom,Face Button Right,Face Button Left,Face Button Top,Left Shoulder,Right Shoulder,Special Left,Special Right,Left Stick Button,Right Stick Button,D-Pad Up,D-Pad Right,D-Pad Down,D-Pad Left", ""
    "Gamepad ID (*inputs:gamepadId*)", "``uint``", "Gamepad id number starting from 0. Each gamepad will be registered automatically with a unique ID monotonically increasing in the order they are connected. If a gamepad is disconnected, the ID assigned to the remaining gamepad will be adjusted accordingly so the IDs are always continuous and start from 0. Changing this value to a non-existing ID will result in an error prompt in the console and the node will not listen to any gamepad input.", "0"
    "", "Metadata", "*literalOnly* = 1", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Is Pressed (*outputs:isPressed*)", "``bool``", "True if the gamepad button was pressed, False if it was released.", "None"
    "Pressed (*outputs:pressed*)", "``execution``", "When the gamepad element was pressed signal to the graph that execution can continue downstream.", "None"
    "Released (*outputs:released*)", "``execution``", "When the gamepad element was released signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnGamepadInput"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "On Gamepad Input"
    "Categories", "graph:action,input:gamepad"
    "Generated Class Name", "OgnOnGamepadInputDatabase"
    "Python Module", "omni.graph.action_nodes"

