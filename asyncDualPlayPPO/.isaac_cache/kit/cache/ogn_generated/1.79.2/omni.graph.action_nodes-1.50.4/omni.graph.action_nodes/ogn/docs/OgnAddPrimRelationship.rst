.. _omni_graph_action_AddPrimRelationship_1:

.. _omni_graph_action_AddPrimRelationship:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Add Prim Relationship
    :keywords: lang-en omnigraph node sceneGraph WriteOnly action add-prim-relationship


Add Prim Relationship
=====================

.. <description>

Adds a target path to a relationship property. If the relationship property does not exist it will be created. Duplicate targets will not be added.

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
    "Relationship Name (*inputs:name*)", "``token``", "Name of the relationship property to be modified or added.", ""
    "Prim Path (*inputs:path*)", "``path``", "Path of the prim with the relationship property.", ""
    "Target Path (*inputs:target*)", "``path``", "The target path to be added, which may be a prim, attribute or another relationship.", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Is Successful (*outputs:isSuccessful*)", "``bool``", "Whether the node has successfully added the new target to the relationship.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.AddPrimRelationship"
    "Version", "1"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Add Prim Relationship"
    "Categories", "sceneGraph"
    "Generated Class Name", "OgnAddPrimRelationshipDatabase"
    "Python Module", "omni.graph.action_nodes"

