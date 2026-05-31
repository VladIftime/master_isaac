.. _omni_graph_action_SwitchToken_2:

.. _omni_graph_action_SwitchToken:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Switch On Token
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action switch-token


Switch On Token
===============

.. <description>

Selectively activates a downstream graph based on the 'Value' name and dynamic attributes. There will be both input and output dynamic attributes for proper functioning of this node. The input attributes will be named 'inputs:branchX', where 'X' is any string, typically just a number.
These input attributes must be a 'token' type, where the value of the token will be compared against the 'Value' input. When the strings match, a corresponding output attribute named 'outputs:outputX' will be checked. If it exists and is of type 'execution' then it will be activated to signal that its downstream graph is ready to be executed. If no matches are found then the node will complete execution without activating any outputs.
For example if 'Value' is set to 'A', and the dynamic attribute 'inputs:branch0' is set to the value 'A' then when this node executes it will active the graph downstream of the dynamic attribute 'outputs:output0'. If multiple branch input attributes contain the same matching value only one of their corresponding outputs will be activated. There is no guarantee as to which of those outputs will be chosen so this situation should be avoided.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"
    "Value (*inputs:value*)", "``token``", "The value to check for in the dynamic input branch attributes.", ""


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.SwitchToken"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Switch On Token"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnSwitchTokenDatabase"
    "Python Module", "omni.graph.action_nodes"

