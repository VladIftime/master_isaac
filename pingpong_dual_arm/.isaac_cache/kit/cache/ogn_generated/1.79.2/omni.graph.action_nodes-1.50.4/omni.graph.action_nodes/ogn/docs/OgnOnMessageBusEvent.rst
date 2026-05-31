.. _omni_graph_action_OnMessageBusEvent_2:

.. _omni_graph_action_OnMessageBusEvent:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On MessageBus Event
    :keywords: lang-en omnigraph node graph:action,event compute-on-request action on-message-bus-event


On MessageBus Event
===================

.. <description>

Event node which fires when the specified event appears on the Application Message Bus. The node uses a callback to monitor messages on the message bus and sets an internal state value when an event named 'Event Name' was received.

.. </description>


Here's an example of sending a kit application message bus event:

.. code-block:: python

    import carb.events
    import omni.kit.app

    msg = carb.events.type_from_string("my_event_name")
    omni.kit.app.get_app().get_message_bus_event_stream().push(msg, payload={ "arg1": 42 })

The event payload data will be copied in to matching dynamic output attributes if they exist.
In the previous example, 42 would be copied to outputs:arg1 if possible. :ref:`SendMessageBusEvent<omni_graph_action_SendMessageBusEvent>` is related.


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Event Name (*inputs:eventName*)", "``token``", "The name of the custom event.", ""
    "", "Metadata", "*literalOnly* = 1", ""
    "Only Simulate On Play (*inputs:onlyPlayback*)", "``bool``", "When true, the node is only executed while the Stage is being played.", "True"
    "", "Metadata", "*literalOnly* = 1", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Received (*outputs:execOut*)", "``execution``", "After receipt of the named message on the application message bus signal to the graph that the message bus event was received and execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.OnMessageBusEvent"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On MessageBus Event"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnOnMessageBusEventDatabase"
    "Python Module", "omni.graph.action_nodes"

