.. _omni_graph_action_OnMouseInput_2:

.. _omni_graph_action_OnMouseInput:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Mouse Input
    :keywords: lang-en omnigraph node graph:action,input:mouse threadsafe compute-on-request action on-mouse-input


On Mouse Input
==============

.. <description>

Event node which fires when a mouse event occurs.
You can choose which 'Mouse Element' this node reacts to. When 'Mouse Element' is chosen to be a button, the only meaningful outputs are: 'Pressed', 'Released' and 'Is Pressed'. When scroll or move events are chosen, the only meaningful outputs are: 'Value Changed' and 'Value'. You can choose to output normalized or pixel coordinates of the mouse.
Pixel coordinates are the absolute position of the mouse cursor in pixel units. The original point is the upper left corner. The minimum value is 0, and the maximum value depends on the size of the window.
Normalized coordinates are the relative position of the mouse cursor to the window. The value is always between 0 and 1.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Mouse Element (*inputs:mouseElement*)", "``token``", "The name of the mouse event that will trigger the downstream execution.", "Left Button"
    "", "Metadata", "*displayGroup* = parameters", ""
    "", "Metadata", "*literalOnly* = 1", ""
    "", "Metadata", "*allowedTokens* = Left Button,Middle Button,Right Button,Normalized Move,Pixel Move,Scroll", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Is Pressed (*outputs:isPressed*)", "``bool``", "True if the mouse button was pressed, False if it was released or 'Mouse Element' is not related to a button, as in a move or scroll.", "None"
    "Pressed (*outputs:pressed*)", "``execution``", "When any mouse button was pressed signal to the graph that the execution can continue downstream. Will not execute on move or scroll events.", "None"
    "Released (*outputs:released*)", "``execution``", "When any mouse button was released signal to the graph that the execution can continue downstream. Will not execute on move or scroll events.", "None"
    "Delta Value (*outputs:value*)", "``float[2]``", "The meaning of this output depends on 'Mouse Element'.  Normalized Move: will output the normalized coordinates of mouse, each element of the vector is in the range of [0, 1].  Pixel Move: will output the absolute coordinates of mouse, each vector is in the range of [0, pixel width/height of the window].  Scroll: will output the change of scroll value.  Otherwise: will output [0,0].", "None"
    "Moved (*outputs:valueChanged*)", "``execution``", "When the mouse was moved or scrolled signal to the graph that the execution can continue downstream. Will not execute on press or release events.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnMouseInput"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Mouse Input"
    "Categories", "graph:action,input:mouse"
    "Generated Class Name", "OgnOnMouseInputDatabase"
    "Python Module", "omni.graph.action_nodes"

