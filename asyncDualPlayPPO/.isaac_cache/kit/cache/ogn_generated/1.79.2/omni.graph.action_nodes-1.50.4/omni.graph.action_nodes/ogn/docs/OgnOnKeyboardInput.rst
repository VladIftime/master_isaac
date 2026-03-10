.. _omni_graph_action_OnKeyboardInput_4:

.. _omni_graph_action_OnKeyboardInput:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Keyboard Input
    :keywords: lang-en omnigraph node graph:action,input:keyboard compute-on-request action on-keyboard-input


On Keyboard Input
=================

.. <description>

Event node which fires when a keyboard event occurs. The event can be any of the keys accepted by 'Key In', plus any combination of modifiers as specified by inputs 'Shift', 'Alt', and 'Ctrl'.
For key combinations, the press event requires all modifiers to be held, with the 'Key In' pressed last. The release event is only triggered once when one of the chosen keys released after the pressed event happens.
For example: if the combination is Ctrl-Shift-D, the pressed event happens once right after D is pressed while both Ctrl and Shift are held. The release event happens only once, when the user releases any one of Ctrl, Shift and D while holding them.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Alt (*inputs:altIn*)", "``bool``", "When true, the Alt key modifier must be pressed along with the 'Key In' to activate the output.", "False"
    "", "Metadata", "*literalOnly* = 1", ""
    "Ctrl (*inputs:ctrlIn*)", "``bool``", "When true, the Ctrl key modifier must be pressed along with the 'Key In' to activate the output.", "False"
    "", "Metadata", "*literalOnly* = 1", ""
    "Key In (*inputs:keyIn*)", "``token``", "The key that triggers the downstream execution, not including modifiers.", "A"
    "", "Metadata", "*displayGroup* = parameters", ""
    "", "Metadata", "*literalOnly* = 1", ""
    "", "Metadata", "*allowedTokens* = A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,Apostrophe,Backslash,Backspace,CapsLock,Comma,Del,Down,End,Enter,Equal,Escape,F1,F10,F11,F12,F2,F3,F4,F5,F6,F7,F8,F9,GraveAccent,Home,Insert,Key0,Key1,Key2,Key3,Key4,Key5,Key6,Key7,Key8,Key9,Left,LeftAlt,LeftBracket,LeftControl,LeftShift,LeftSuper,Menu,Minus,NumLock,Numpad0,Numpad1,Numpad2,Numpad3,Numpad4,Numpad5,Numpad6,Numpad7,Numpad8,Numpad9,NumpadAdd,NumpadDel,NumpadDivide,NumpadEnter,NumpadEqual,NumpadMultiply,NumpadSubtract,PageDown,PageUp,Pause,Period,PrintScreen,Right,RightAlt,RightBracket,RightControl,RightShift,RightSuper,ScrollLock,Semicolon,Slash,Space,Tab,Up", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""
    "Shift (*inputs:shiftIn*)", "``bool``", "When true, the Shift key modifier must be pressed along with the 'Key In' to activate the output.", "False"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Is Pressed (*outputs:isPressed*)", "``bool``", "True if the most recent activation was the key being pressed, False if it was released.", "None"
    "Key Out (*outputs:keyOut*)", "``token``", "The key that was pressed or released to trigger the execution of this node.", "None"
    "Pressed (*outputs:pressed*)", "``execution``", "After the key was pressed, along with any required modifiers, signal to the graph that the execution can continue downstream.", "None"
    "Released (*outputs:released*)", "``execution``", "After the key or any of the required modifiers were released, signal to the graph that the execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnKeyboardInput"
    "Version", "4"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Keyboard Input"
    "Categories", "graph:action,input:keyboard"
    "Generated Class Name", "OgnOnKeyboardInputDatabase"
    "Python Module", "omni.graph.action_nodes"

