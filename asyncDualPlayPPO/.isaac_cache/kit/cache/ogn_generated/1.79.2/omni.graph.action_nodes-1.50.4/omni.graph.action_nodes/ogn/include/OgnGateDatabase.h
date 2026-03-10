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

namespace OgnGateAttributes
{
namespace inputs
{
using enter_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> enter("inputs:enter", "execution", kExtendedAttributeType_Regular);
using startClosed_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> startClosed("inputs:startClosed", "bool", kExtendedAttributeType_Regular, false);
using toggle_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> toggle("inputs:toggle", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using exit_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> exit("outputs:exit", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnGateAttributes;
namespace IOgnGate
{
// This node controls a flow of execution based on the state of its gate. The gate can
// be opened or closed by activation of the 'Toggle' gate controls. Each time 'Enter'
// is activated, the node will activate the 'Exit' signal if the gate is open, and silently
// succeed if the gate is closed. The current state of the gate is not directly accessible.
class OgnGateDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnGate.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnGate.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnGate.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnGate.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnGate;
    static std::tuple<int, int, int>sm_generatorVersionOgnGate;
    static std::tuple<int, int, int>sm_targetVersionOgnGate;
    static constexpr size_t staticAttributeCount = 6;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : enter{offset,AttributeRole::eExecution}
        , startClosed{offset}
        , toggle{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> enter;
        ogn::SimpleInput<const bool,ogn::kCpu> startClosed;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> toggle;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : exit{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> exit;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnGateDatabase(NodeObj const& nodeObjParam)
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
    OgnGateDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnGateDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnGateDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::enter.m_token, inputs::startClosed.m_token, inputs::toggle.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::exit.m_token
                )
            , kAccordingToContextIndex);
            inputs.enter.setContext(contextObj);
            inputs.enter.setHandle(std::get<0>(inputDataHandles0));
            inputs.startClosed.setContext(contextObj);
            inputs.startClosed.setHandle(std::get<1>(inputDataHandles0));
            inputs.toggle.setContext(contextObj);
            inputs.toggle.setHandle(std::get<2>(inputDataHandles0));
            outputs.exit.setContext(contextObj);
            outputs.exit.setHandle(std::get<0>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.Gate");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::enter.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::startClosed.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::toggle.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::exit.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Gate");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "This node controls a flow of execution based on the state of its gate. The gate can be opened or closed by activation of the 'Toggle' gate controls. Each time 'Enter' is activated, the node will activate the 'Exit' signal if the gate is open, and silently succeed if the gate is closed. The current state of the gate is not directly accessible.");
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::enter.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.\nBefore this signal is activated the gate will be in the state specified by the\n'Start Closed' value.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Enter");
        attr = iNode->getAttributeByToken(nodeObj, inputs::startClosed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "If true the gate will start in a closed state.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Start Closed");
        attr = iNode->getAttributeByToken(nodeObj, inputs::toggle.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the node that the state of the gate should switch from open to closed,\nor vice versa. This signal will not activate the 'Exit', it will only determine whether\nor not the next 'Enter' will activate it.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Toggle");
        attr = iNode->getAttributeByToken(nodeObj, outputs::exit.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When 'Enter' is activated and the gate is open signal to the graph that execution\nshould continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Exit");
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
        sm_stateManagerOgnGate.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.enter.isValid()
            && inputs.startClosed.isValid()
            && inputs.toggle.isValid()
            && outputs.exit.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.enter.invalidateCachedPointer();
            inputs.startClosed.invalidateCachedPointer();
            inputs.toggle.invalidateCachedPointer();
            outputs.exit.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::enter.m_token) {
                inputs.enter.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::startClosed.m_token) {
                inputs.startClosed.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::toggle.m_token) {
                inputs.toggle.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::exit.m_token) {
                outputs.exit.invalidateCachedPointer();
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
        if( !inputs.enter.canVectorize()
            || !inputs.startClosed.canVectorize()
            || !inputs.toggle.canVectorize()
            || !outputs.exit.canVectorize()
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
        if(token == inputs::enter.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.enter.setHandle(hdl);
            return;
        }
        if(token == inputs::startClosed.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.startClosed.setHandle(hdl);
            return;
        }
        if(token == inputs::toggle.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.toggle.setHandle(hdl);
            return;
        }
        if(token == outputs::exit.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.exit.setHandle(hdl);
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
ogn::StateManager OgnGateDatabase::sm_stateManagerOgnGate;
std::tuple<int, int, int> OgnGateDatabase::sm_generatorVersionOgnGate{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnGateDatabase::sm_targetVersionOgnGate{std::make_tuple(2,184,5)};
}
using namespace IOgnGate;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnGate, OgnGateDatabase> s_registration("omni.graph.action.Gate", 2, "omni.graph.action_nodes"); \
}
