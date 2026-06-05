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

namespace OgnFlipFlopAttributes
{
namespace inputs
{
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using a_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> a("outputs:a", "execution", kExtendedAttributeType_Regular);
using b_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> b("outputs:b", "execution", kExtendedAttributeType_Regular);
using isA_t = bool&;
ogn::AttributeInitializer<bool, ogn::kOgnOutput> isA("outputs:isA", "bool", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnFlipFlopAttributes;
namespace IOgnFlipFlop
{
// This node activates its outputs in an alternating sequence, starting with 'On Odd'
// on the first execution after 'Signal Execution' has been activated.
class OgnFlipFlopDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnFlipFlop.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnFlipFlop.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnFlipFlop.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnFlipFlop.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnFlipFlop;
    static std::tuple<int, int, int>sm_generatorVersionOgnFlipFlop;
    static std::tuple<int, int, int>sm_targetVersionOgnFlipFlop;
    static constexpr size_t staticAttributeCount = 6;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : execIn{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : a{offset,AttributeRole::eExecution}
        , b{offset,AttributeRole::eExecution}
        , isA{offset}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> a;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> b;
        ogn::SimpleOutput<bool,ogn::kCpu> isA;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnFlipFlopDatabase(NodeObj const& nodeObjParam)
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
    OgnFlipFlopDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnFlipFlopDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnFlipFlopDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::execIn.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::a.m_token, outputs::b.m_token, outputs::isA.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            outputs.a.setContext(contextObj);
            outputs.a.setHandle(std::get<0>(outputDataHandles0));
            outputs.b.setContext(contextObj);
            outputs.b.setHandle(std::get<1>(outputDataHandles0));
            outputs.isA.setContext(contextObj);
            outputs.isA.setHandle(std::get<2>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.FlipFlop");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::a.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::b.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::isA.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Flip Flop");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "This node activates its outputs in an alternating sequence, starting with 'On Odd' on the first execution after 'Signal Execution' has been activated.");
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
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute In");
        attr = iNode->getAttributeByToken(nodeObj, outputs::a.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After every odd-numbered execution signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute A");
        attr = iNode->getAttributeByToken(nodeObj, outputs::b.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After every even-numbered execution signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute B");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isA.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Set to true when the 'Execute A' signal is active, otherwise set to false.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Is A");
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
        sm_stateManagerOgnFlipFlop.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.execIn.isValid()
            && outputs.a.isValid()
            && outputs.b.isValid()
            && outputs.isA.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.execIn.invalidateCachedPointer();
            outputs.a.invalidateCachedPointer();
            outputs.b.invalidateCachedPointer();
            outputs.isA.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::a.m_token) {
                outputs.a.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::b.m_token) {
                outputs.b.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::isA.m_token) {
                outputs.isA.invalidateCachedPointer();
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
            || !outputs.a.canVectorize()
            || !outputs.b.canVectorize()
            || !outputs.isA.canVectorize()
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
        if(token == outputs::a.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.a.setHandle(hdl);
            return;
        }
        if(token == outputs::b.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.b.setHandle(hdl);
            return;
        }
        if(token == outputs::isA.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.isA.setHandle(hdl);
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
ogn::StateManager OgnFlipFlopDatabase::sm_stateManagerOgnFlipFlop;
std::tuple<int, int, int> OgnFlipFlopDatabase::sm_generatorVersionOgnFlipFlop{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnFlipFlopDatabase::sm_targetVersionOgnFlipFlop{std::make_tuple(2,184,5)};
}
using namespace IOgnFlipFlop;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnFlipFlop, OgnFlipFlopDatabase> s_registration("omni.graph.action.FlipFlop", 2, "omni.graph.action_nodes"); \
}
