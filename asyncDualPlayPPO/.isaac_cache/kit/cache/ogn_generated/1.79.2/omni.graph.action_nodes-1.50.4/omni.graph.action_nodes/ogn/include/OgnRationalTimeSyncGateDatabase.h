#pragma once

#include <omni/graph/core/ISchedulingHints2.h>
#include <carb/InterfaceUtils.h>
#include <omni/graph/core/NodeTypeRegistrar.h>
#include <omni/graph/core/iComputeGraph.h>
#include <omni/graph/core/CppWrappers.h>
#include <omni/fabric/Enums.h>
using omni::fabric::PtrToPtrKind;
#include <map>
#include <vector>
#include <tuple>
#include <omni/graph/core/OgnHelpers.h>
#include <omni/graph/core/Type.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>

namespace OgnRationalTimeSyncGateAttributes
{
namespace inputs
{
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using rationalTimeDenominator_t = const uint64_t&;
ogn::AttributeInitializer<const uint64_t, ogn::kOgnInput> rationalTimeDenominator("inputs:rationalTimeDenominator", "uint64", kExtendedAttributeType_Regular, 0);
using rationalTimeNumerator_t = const int64_t&;
ogn::AttributeInitializer<const int64_t, ogn::kOgnInput> rationalTimeNumerator("inputs:rationalTimeNumerator", "int64", kExtendedAttributeType_Regular, 0);
}
namespace outputs
{
using execOut_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> execOut("outputs:execOut", "execution", kExtendedAttributeType_Regular);
using rationalTimeDenominator_t = uint64_t&;
ogn::AttributeInitializer<uint64_t, ogn::kOgnOutput> rationalTimeDenominator("outputs:rationalTimeDenominator", "uint64", kExtendedAttributeType_Regular);
using rationalTimeNumerator_t = int64_t&;
ogn::AttributeInitializer<int64_t, ogn::kOgnOutput> rationalTimeNumerator("outputs:rationalTimeNumerator", "int64", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnRationalTimeSyncGateAttributes;
namespace IOgnRationalTimeSyncGate
{
// Activate the downstream graphs after all of the input signals have been triggered
// with the same synchronization value. An internal count is maintained that resets
// whenever a new synchronization value is encountered. If the count reaches a number
// equal to the number of 'Execute In' inputs then the 'Execute Out' activation signals
// to the downstream graph that it is ready to be executed.
// The synchronization value is expressed as a rational number, conceptually equal to
// 'Sync Numerator' / 'Sync Denominator', although a zero denominator is accepted because
// the division is not actually performed. The comparison is made between two rational
// values by first reducing them by their greatest common denominators and then comparing
// numerator and denominator individually.
// This means, for example, that 1/2 and 3/6 will be considered the same, but 1/0 and
// 2/0 will not since they cannot be reduced.
class OgnRationalTimeSyncGateDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnRationalTimeSyncGate.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnRationalTimeSyncGate.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnRationalTimeSyncGate.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnRationalTimeSyncGate.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
    }
    template <typename StateInformation>
    CARB_DEPRECATED("internalState is deprecated. Use sharedState or perInstanceState instead")
    StateInformation& internalState(size_t relativeIdx = 0) {
        return sInternalState<StateInformation>(abi_node(), m_offset + relativeIdx);
    }
    template <typename StateInformation>
    StateInformation& sharedState() {
        return sSharedState<StateInformation>(abi_node());
    }
    template <typename StateInformation>
    StateInformation& perInstanceState(size_t relativeIdx = 0) {
        return sPerInstanceState<StateInformation>(abi_node(), m_offset + relativeIdx);
    }
    template <typename StateInformation>
    StateInformation& perInstanceState(GraphInstanceID instanceId) {
        return sPerInstanceState<StateInformation>(abi_node(), instanceId);
    }
    static ogn::StateManager sm_stateManagerOgnRationalTimeSyncGate;
    static std::tuple<int, int, int>sm_generatorVersionOgnRationalTimeSyncGate;
    static std::tuple<int, int, int>sm_targetVersionOgnRationalTimeSyncGate;
    static constexpr size_t staticAttributeCount = 8;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : execIn{offset,AttributeRole::eExecution}
        , rationalTimeDenominator{offset}
        , rationalTimeNumerator{offset}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::SimpleInput<const uint64_t,ogn::kCpu> rationalTimeDenominator;
        ogn::SimpleInput<const int64_t,ogn::kCpu> rationalTimeNumerator;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : execOut{offset,AttributeRole::eExecution}
        , rationalTimeDenominator{offset}
        , rationalTimeNumerator{offset}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> execOut;
        ogn::SimpleOutput<uint64_t,ogn::kCpu> rationalTimeDenominator;
        ogn::SimpleOutput<int64_t,ogn::kCpu> rationalTimeNumerator;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnRationalTimeSyncGateDatabase(NodeObj const& nodeObjParam)
    : OmniGraphDatabase()
    , inputs{m_offset.index}
    , outputs{m_offset.index}
    {
        GraphContextObj const* contexts = nullptr;
        NodeObj const* nodes = nullptr;
        size_t handleCount = nodeObjParam.iNode->getAutoInstances(nodeObjParam, contexts, nodes);
        _ctor(contexts, nodes, handleCount);
        _init();
    }

    CARB_DEPRECATED("Passing the graph context to the temporary stack allocated database is not necessary anymore: you can safely remove this parameter")
    OgnRationalTimeSyncGateDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnRationalTimeSyncGateDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnRationalTimeSyncGateDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
    : OmniGraphDatabase()
    , inputs{m_offset.index}
    , outputs{m_offset.index}
    {
        _ctor(contextObjParam, nodeObjParam, handleCount);
        _init();
    }

private:
    void _init() {
        GraphContextObj const& contextObj = abi_context();
        NodeObj const& nodeObj = abi_node();
        {
            auto inputDataHandles0 = getAttributesR<
                ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::execIn.m_token, inputs::rationalTimeDenominator.m_token, inputs::rationalTimeNumerator.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::execOut.m_token, outputs::rationalTimeDenominator.m_token, outputs::rationalTimeNumerator.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.rationalTimeDenominator.setContext(contextObj);
            inputs.rationalTimeDenominator.setHandle(std::get<1>(inputDataHandles0));
            inputs.rationalTimeNumerator.setContext(contextObj);
            inputs.rationalTimeNumerator.setHandle(std::get<2>(inputDataHandles0));
            outputs.execOut.setContext(contextObj);
            outputs.execOut.setHandle(std::get<0>(outputDataHandles0));
            outputs.rationalTimeDenominator.setContext(contextObj);
            outputs.rationalTimeDenominator.setHandle(std::get<1>(outputDataHandles0));
            outputs.rationalTimeNumerator.setContext(contextObj);
            outputs.rationalTimeNumerator.setHandle(std::get<2>(outputDataHandles0));
        }
        tryGetDynamicAttributes<AttributePortType::kAttributePortType_Input>(staticAttributeCount, m_dynamicInputs);
        tryGetDynamicAttributes<AttributePortType::kAttributePortType_Output>(staticAttributeCount, m_dynamicOutputs);
        tryGetDynamicAttributes<AttributePortType::kAttributePortType_State>(staticAttributeCount, m_dynamicStates);

        collectMappedAttributes(m_mappedAttributes);

        m_canCachePointers = contextObj.iContext->canCacheAttributePointers ?
                                 contextObj.iContext->canCacheAttributePointers(contextObj) : true;
    }

public:
    static void initializeType(const NodeTypeObj& nodeTypeObj)
    {
        const INodeType* iNodeType = nodeTypeObj.iNodeType;
        auto iTokenPtr = carb::getCachedInterface<omni::fabric::IToken>();
        if( ! iTokenPtr ) {
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.RationalTimeSyncGate");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::rationalTimeDenominator.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::rationalTimeNumerator.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::execOut.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::rationalTimeDenominator.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::rationalTimeNumerator.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Rational Sync Gate");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Activate the downstream graphs after all of the input signals have been triggered with the same synchronization value. An internal count is maintained that resets whenever a new synchronization value is encountered. If the count reaches a number equal to the number of 'Execute In' inputs then the 'Execute Out' activation signals to the downstream graph that it is ready to be executed.\nThe synchronization value is expressed as a rational number, conceptually equal to 'Sync Numerator' / 'Sync Denominator', although a zero denominator is accepted because the division is not actually performed. The comparison is made between two rational values by first reducing them by their greatest common denominators and then comparing numerator and denominator individually.\nThis means, for example, that 1/2 and 3/6 will be considered the same, but 1/0 and 2/0 will not since they cannot be reduced.");
        auto __schedulingInfo = nodeTypeObj.iNodeType->getSchedulingHints(nodeTypeObj);
        CARB_ASSERT(__schedulingInfo, "Could not acquire the scheduling hints");
        if (__schedulingInfo)
        {
            __schedulingInfo->setThreadSafety(eThreadSafety::eSafe);
            auto __schedulingInfo2 = omni::core::cast<ISchedulingHints2>(__schedulingInfo).get();
            if (__schedulingInfo2)
            {
            }
        }
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute In");
        attr = iNode->getAttributeByToken(nodeObj, inputs::rationalTimeDenominator.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Denominator of the synchronization time.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Sync Denominator");
        attr = iNode->getAttributeByToken(nodeObj, inputs::rationalTimeNumerator.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Numerator of the synchronization time.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Sync Numerator");
        attr = iNode->getAttributeByToken(nodeObj, outputs::execOut.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After all 'Execute In' connections have been activated at the same rational time, signal\nto the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute Out");
        attr = iNode->getAttributeByToken(nodeObj, outputs::rationalTimeDenominator.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Denominator of the synchronization time, whether 'Execute Out' was activated or not.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Sync Denominator");
        attr = iNode->getAttributeByToken(nodeObj, outputs::rationalTimeNumerator.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Numerator of the synchronization time, whether 'Execute Out' was activated or not.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Sync Numerator");
    }
    std::vector<ogn::DynamicInput> const& getDynamicInputs() const
    {
        return m_dynamicInputs;
    }
    gsl::span<ogn::DynamicOutput> getDynamicOutputs()
    {
        return m_dynamicOutputs;
    }
    gsl::span<ogn::DynamicState> getDynamicStates()
    {
        return m_dynamicStates;
    }
    static void release(const NodeObj& nodeObj, GraphInstanceID instanceID)
    {
        sm_stateManagerOgnRationalTimeSyncGate.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.execIn.isValid()
            && inputs.rationalTimeDenominator.isValid()
            && inputs.rationalTimeNumerator.isValid()
            && outputs.execOut.isValid()
            && outputs.rationalTimeDenominator.isValid()
            && outputs.rationalTimeNumerator.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.execIn.invalidateCachedPointer();
            inputs.rationalTimeDenominator.invalidateCachedPointer();
            inputs.rationalTimeNumerator.invalidateCachedPointer();
            outputs.execOut.invalidateCachedPointer();
            outputs.rationalTimeDenominator.invalidateCachedPointer();
            outputs.rationalTimeNumerator.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::rationalTimeDenominator.m_token) {
                inputs.rationalTimeDenominator.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::rationalTimeNumerator.m_token) {
                inputs.rationalTimeNumerator.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::execOut.m_token) {
                outputs.execOut.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::rationalTimeDenominator.m_token) {
                outputs.rationalTimeDenominator.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::rationalTimeNumerator.m_token) {
                outputs.rationalTimeNumerator.invalidateCachedPointer();
                continue;
            }
            bool found = false;
            for (auto& __a : m_dynamicInputs) {
                if (__a().name() == attrib) {
                    __a.invalidateCachedPointer();
                    found = true;
                    break;
                }
            }
            if(found) continue;
            for (auto& __a : m_dynamicOutputs) {
                if (__a().name() == attrib) {
                    __a.invalidateCachedPointer();
                    found = true;
                    break;
                }
            }
            if(found) continue;
            for (auto& __a : m_dynamicStates) {
                if (__a().name() == attrib) {
                    __a.invalidateCachedPointer();
                    found = true;
                    break;
                }
            }
            if(found) continue;
        }
    }
    bool canVectorize() const {
        if( !inputs.execIn.canVectorize()
            || !inputs.rationalTimeDenominator.canVectorize()
            || !inputs.rationalTimeNumerator.canVectorize()
            || !outputs.execOut.canVectorize()
            || !outputs.rationalTimeDenominator.canVectorize()
            || !outputs.rationalTimeNumerator.canVectorize()
        ) return false;
        for (auto const& __a : m_dynamicInputs) {
            if(!__a.canVectorize()) return false;
        }
        for (auto const& __a : m_dynamicOutputs) {
            if(!__a.canVectorize()) return false;
        }
        for (auto const& __a : m_dynamicStates) {
            if(!__a.canVectorize()) return false;
        }
        return true;
    }
    void onTypeResolutionChanged(AttributeObj const& attr) {
        if (! attr.isValid()) return;
        NameToken token = attr.iAttribute->getNameToken(attr);
        for (auto& __a : m_dynamicInputs) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
        for (auto& __a : m_dynamicOutputs) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
        for (auto& __a : m_dynamicStates) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
    }
    void onDynamicAttributesChanged(AttributeObj const& attribute, bool isAttributeCreated) {
        onDynamicAttributeCreatedOrRemoved(m_dynamicInputs, m_dynamicOutputs, m_dynamicStates, attribute, isAttributeCreated);
    }
    void onDataLocationChanged(AttributeObj const& attr) {
        if (! attr.isValid()) return;
        updateMappedAttributes(m_mappedAttributes, attr);
        NameToken token = attr.iAttribute->getNameToken(attr);
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == inputs::rationalTimeDenominator.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.rationalTimeDenominator.setHandle(hdl);
            return;
        }
        if(token == inputs::rationalTimeNumerator.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.rationalTimeNumerator.setHandle(hdl);
            return;
        }
        if(token == outputs::execOut.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.execOut.setHandle(hdl);
            return;
        }
        if(token == outputs::rationalTimeDenominator.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.rationalTimeDenominator.setHandle(hdl);
            return;
        }
        if(token == outputs::rationalTimeNumerator.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.rationalTimeNumerator.setHandle(hdl);
            return;
        }
        for (auto& __a : m_dynamicInputs) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
        for (auto& __a : m_dynamicOutputs) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
        for (auto& __a : m_dynamicStates) {
            if (__a().name() == token) {
                __a.fetchDetails(attr);
                return;
            }
        }
    }
};
ogn::StateManager OgnRationalTimeSyncGateDatabase::sm_stateManagerOgnRationalTimeSyncGate;
std::tuple<int, int, int> OgnRationalTimeSyncGateDatabase::sm_generatorVersionOgnRationalTimeSyncGate{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnRationalTimeSyncGateDatabase::sm_targetVersionOgnRationalTimeSyncGate{std::make_tuple(2,184,5)};
}
using namespace IOgnRationalTimeSyncGate;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnRationalTimeSyncGate, OgnRationalTimeSyncGateDatabase> s_registration("omni.graph.action.RationalTimeSyncGate", 2, "omni.graph.action_nodes"); \
}
