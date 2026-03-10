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

namespace OgnCounterAttributes
{
namespace inputs
{
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using reset_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> reset("inputs:reset", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using count_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnOutput> count("outputs:count", "int", kExtendedAttributeType_Regular);
using execOut_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> execOut("outputs:execOut", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
using count_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnState> count("state:count", "int", kExtendedAttributeType_Regular);
}
}
using namespace OgnCounterAttributes;
namespace IOgnCounter
{
// This node counts the number of times it has been executed since the 'Reset' signal
// was activated.
class OgnCounterDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnCounter.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnCounter.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnCounter.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnCounter.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnCounter;
    static std::tuple<int, int, int>sm_generatorVersionOgnCounter;
    static std::tuple<int, int, int>sm_targetVersionOgnCounter;
    static constexpr size_t staticAttributeCount = 7;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : execIn{offset,AttributeRole::eExecution}
        , reset{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> reset;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : count{offset}
        , execOut{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<int,ogn::kCpu> count;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> execOut;
    } outputs;

    struct stateT {
        stateT(size_t const& offset)
        : count{offset}
        {}
        ogn::SimpleState<int,ogn::kCpu> count;
    } state;

    //Only use this constructor for temporary stack-allocated object:
    OgnCounterDatabase(NodeObj const& nodeObjParam)
    : OmniGraphDatabase()
    , inputs{m_offset.index}
    , outputs{m_offset.index}
    , state{m_offset.index}
    {
        GraphContextObj const* contexts = nullptr;
        NodeObj const* nodes = nullptr;
        size_t handleCount = nodeObjParam.iNode->getAutoInstances(nodeObjParam, contexts, nodes);
        _ctor(contexts, nodes, handleCount);
        _init();
    }

    CARB_DEPRECATED("Passing the graph context to the temporary stack allocated database is not necessary anymore: you can safely remove this parameter")
    OgnCounterDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnCounterDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnCounterDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
    : OmniGraphDatabase()
    , inputs{m_offset.index}
    , outputs{m_offset.index}
    , state{m_offset.index}
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
                ConstAttributeDataHandle, ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::execIn.m_token, inputs::reset.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::count.m_token, outputs::execOut.m_token
                )
            , kAccordingToContextIndex);
            auto stateDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    state::count.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.reset.setContext(contextObj);
            inputs.reset.setHandle(std::get<1>(inputDataHandles0));
            outputs.count.setContext(contextObj);
            outputs.count.setHandle(std::get<0>(outputDataHandles0));
            outputs.execOut.setContext(contextObj);
            outputs.execOut.setHandle(std::get<1>(outputDataHandles0));
            state.count.setContext(contextObj);
            state.count.setHandle(std::get<0>(stateDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.Counter");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::reset.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::count.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::execOut.initialize(iToken, *iNodeType, nodeTypeObj);

        state::count.initialize(iToken, *iNodeType, nodeTypeObj);
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Counter");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,function");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "This node counts the number of times it has been executed since the 'Reset' signal was activated.");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExclusions, "tests");
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
        iNodeType->setHasState(nodeTypeObj, true);
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute In");
        attr = iNode->getAttributeByToken(nodeObj, inputs::reset.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the node to reset its internal counter.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Reset");
        attr = iNode->getAttributeByToken(nodeObj, outputs::count.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The number of times this node has been executed since being reset.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Count");
        attr = iNode->getAttributeByToken(nodeObj, outputs::execOut.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute Out");
        attr = iNode->getAttributeByToken(nodeObj, state::count.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Internal value storing the execution count in a persistent way.");
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
        sm_stateManagerOgnCounter.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.execIn.isValid()
            && inputs.reset.isValid()
            && outputs.count.isValid()
            && outputs.execOut.isValid()
            && state.count.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.execIn.invalidateCachedPointer();
            inputs.reset.invalidateCachedPointer();
            outputs.count.invalidateCachedPointer();
            outputs.execOut.invalidateCachedPointer();
            state.count.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::reset.m_token) {
                inputs.reset.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::count.m_token) {
                outputs.count.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::execOut.m_token) {
                outputs.execOut.invalidateCachedPointer();
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
            || !inputs.reset.canVectorize()
            || !outputs.count.canVectorize()
            || !outputs.execOut.canVectorize()
            || !state.count.canVectorize()
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
        if(token == inputs::reset.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.reset.setHandle(hdl);
            return;
        }
        if(token == outputs::count.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.count.setHandle(hdl);
            return;
        }
        if(token == outputs::execOut.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.execOut.setHandle(hdl);
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
ogn::StateManager OgnCounterDatabase::sm_stateManagerOgnCounter;
std::tuple<int, int, int> OgnCounterDatabase::sm_generatorVersionOgnCounter{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnCounterDatabase::sm_targetVersionOgnCounter{std::make_tuple(2,184,5)};
}
using namespace IOgnCounter;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnCounter, OgnCounterDatabase> s_registration("omni.graph.action.Counter", 2, "omni.graph.action_nodes"); \
}
