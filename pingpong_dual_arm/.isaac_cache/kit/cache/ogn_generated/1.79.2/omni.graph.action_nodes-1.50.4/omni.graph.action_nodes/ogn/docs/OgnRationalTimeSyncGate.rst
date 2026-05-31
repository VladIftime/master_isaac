.. _omni_graph_action_RationalTimeSyncGate_2:

.. _omni_graph_action_RationalTimeSyncGate:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Rational Sync Gate
    :keywords: lang-en omnigraph node graph:action,flowControl threadsafe action rational-time-sync-gate


Rational Sync Gate
==================

.. <description>

Activate the downstream graphs after all of the input signals have been triggered with the same synchronization value. An internal count is maintained that resets whenever a new synchronization value is encountered. If the count reaches a number equal to the number of 'Execute In' inputs then the 'Execute Out' activation signals to the downstream graph that it is ready to be executed.
The synchronization value is expressed as a rational number, conceptually equal to 'Sync Numerator' / 'Sync Denominator', although a zero denominator is accepted because the division is not actually performed. The comparison is made between two rational values by first reducing them by their greatest common denominators and then comparing numerator and denominator individually.
This means, for example, that 1/2 and 3/6 will be considered the same, but 1/0 and 2/0 will not since they cannot be reduced.

.. </description>


Installation
------------

To use this node enable :ref:`omni.graph.action_nodes<ext_omni_graph_action_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Execute In (*inputs:execIn*)", "``execution``", "Signal to the graph that this node is ready to be executed.", "None"
    "Sync Denominator (*inputs:rationalTimeDenominator*)", "``uint64``", "Denominator of the synchronization time.", "0"
    "Sync Numerator (*inputs:rationalTimeNumerator*)", "``int64``", "Numerator of the synchronization time.", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Execute Out (*outputs:execOut*)", "``execution``", "After all 'Execute In' connections have been activated at the same rational time, signal to the graph that execution can continue downstream.", "None"
    "Sync Denominator (*outputs:rationalTimeDenominator*)", "``uint64``", "Denominator of the synchronization time, whether 'Execute Out' was activated or not.", "None"
    "Sync Numerator (*outputs:rationalTimeNumerator*)", "``int64``", "Numerator of the synchronization time, whether 'Execute Out' was activated or not.", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.graph.action.RationalTimeSyncGate"
    "Version", "2"
    "Extension", "omni.graph.action_nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Rational Sync Gate"
    "Categories", "graph:action,flowControl"
    "Generated Class Name", "OgnRationalTimeSyncGateDatabase"
    "Python Module", "omni.graph.action_nodes"

