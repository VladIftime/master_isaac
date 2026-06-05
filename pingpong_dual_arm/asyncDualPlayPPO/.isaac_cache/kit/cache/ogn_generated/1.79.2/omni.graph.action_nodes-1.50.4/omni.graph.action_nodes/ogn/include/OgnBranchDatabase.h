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

namespace OgnBranchAttributes
{
namespace inputs
{
using condition_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> condition("inputs:condition", "bool", kExtendedAttributeType_Regular, false);
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using execFalse_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> execFalse("outputs:execFalse", "execution", kExtendedAttributeType_Regular);
using execTrue_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> execTrue("outputs:execTrue", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnBranchAttributes;
namespace IOgnBranch
{
// Activates an execution output signal along a branch based on a boolean condition.
class OgnBranchDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnBranch.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnBranch.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnBranch.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnBranch.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnBranch;
    static std::tuple<int, int, int>sm_generatorVersionOgnBranch;
    static std::tuple<int, int, int>sm_targetVersionOgnBranch;
    static constexpr size_t staticAttributeCount = 6;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : condition{offset}
        , execIn{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const bool,ogn::kCpu> condition;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : execFalse{offset,AttributeRole::eExecution}
        , execTrue{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> execFalse;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> execTrue;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnBranchDatabase(NodeObj const& nodeObjParam)
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
    OgnBranchDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnBranchDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnBranchDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle, ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::condition.m_token, inputs::execIn.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::execFalse.m_token, outputs::execTrue.m_token
                )
            , kAccordingToContextIndex);
            inputs.condition.setContext(contextObj);
            inputs.condition.setHandle(std::get<0>(inputDataHandles0));
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<1>(inputDataHandles0));
            outputs.execFalse.setContext(contextObj);
            outputs.execFalse.setHandle(std::get<0>(outputDataHandles0));
            outputs.execTrue.setContext(contextObj);
            outputs.execTrue.setHandle(std::get<1>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.Branch");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::condition.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::execFalse.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::execTrue.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Branch");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Activates an execution output signal along a branch based on a boolean condition.");
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::condition.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The boolean condition which determines the output direction.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Condition");
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Input execution");
        attr = iNode->getAttributeByToken(nodeObj, outputs::execFalse.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When 'Condition' is False signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "False");
        attr = iNode->getAttributeByToken(nodeObj, outputs::execTrue.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When 'Condition' is True signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "True");
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
        sm_stateManagerOgnBranch.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.condition.isValid()
            && inputs.execIn.isValid()
            && outputs.execFalse.isValid()
            && outputs.execTrue.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.condition.invalidateCachedPointer();
            inputs.execIn.invalidateCachedPointer();
            outputs.execFalse.invalidateCachedPointer();
            outputs.execTrue.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::condition.m_token) {
                inputs.condition.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::execFalse.m_token) {
                outputs.execFalse.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::execTrue.m_token) {
                outputs.execTrue.invalidateCachedPointer();
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
        if( !inputs.condition.canVectorize()
            || !inputs.execIn.canVectorize()
            || !outputs.execFalse.canVectorize()
            || !outputs.execTrue.canVectorize()
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
        if(token == inputs::condition.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.condition.setHandle(hdl);
            return;
        }
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == outputs::execFalse.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.execFalse.setHandle(hdl);
            return;
        }
        if(token == outputs::execTrue.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.execTrue.setHandle(hdl);
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
ogn::StateManager OgnBranchDatabase::sm_stateManagerOgnBranch;
std::tuple<int, int, int> OgnBranchDatabase::sm_generatorVersionOgnBranch{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnBranchDatabase::sm_targetVersionOgnBranch{std::make_tuple(2,184,5)};
}
using namespace IOgnBranch;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnBranch, OgnBranchDatabase> s_registration("omni.graph.action.Branch", 2, "omni.graph.action_nodes"); \
}
