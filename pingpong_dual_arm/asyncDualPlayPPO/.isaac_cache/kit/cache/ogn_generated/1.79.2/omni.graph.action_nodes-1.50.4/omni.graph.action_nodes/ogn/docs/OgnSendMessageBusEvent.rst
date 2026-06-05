.. _omni_graph_action_SendMessageBusEvent_1:

.. _omni_graph_action_SendMessageBusEvent:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Send MessageBus Event
    :keywords: lang-en omnigraph node graph:action,event threadsafe action send-message-bus-event


Send MessageBus Event
=====================

.. <description>

Pushes a named event to the Application Message Bus. The payload of the event is constructed from any dynamic input attributes that have been added to this node and consists of a dictionary whose key is the name of the attribute, with the 'inputs:' prefix stripped from it, with a value equal to the raw pointer to the attribute's data in Fabric. If the attribute has a type that is not able to be converted to this form then a warning is posted for the node but the execution continues.

.. </description>

The event can be handled by any message bus listener, or with a corresponding :ref:`OnMessageBusEvent<omni_graph_action_OnMessageBusEvent>` node.

Data from dynamic input attributes will be copied into the event payload, with keys that match the attribute name. Here's an example of handling the event in Python. In this example the sending node has an input "inputs:arg1":

.. code-block:: python

    import carb.events
    import omni.kit.app

    def on_event(event: carb.events.IEvent):
        data = event.payload["arg1"]
        print(f"got data = {data}")

    msg = carb.events.type_from_string("my_event_name")
    message_bus = omni.kit.app.get_app().get_message_bus_event_stream()
    sub = message_bus.create_subscription_to_pop_by_type(msg, on_event)


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Event Name (*inputs:eventName*)", "``token``", "The name of the custom event to be sent.", ""
    "", "Metadata", "*literalOnly* = 1", ""
    "Exec In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "Signal to the graph that execution can continue downstream.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.SendMessageBusEvent"
    "Version", "1"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "Send MessageBus Event"
    "Categories", "graph:action,event"
    "Generated Class Name", "OgnSendMessageBusEventDatabase"
    "Python Module", "omni.graph.action_nodes"

