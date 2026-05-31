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

namespace OgnDelayAttributes
{
namespace inputs
{
using duration_t = const double&;
ogn::AttributeInitializer<const double, ogn::kOgnInput> duration("inputs:duration", "double", kExtendedAttributeType_Regular, 0.0);
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using finished_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> finished("outputs:finished", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnDelayAttributes;
namespace IOgnDelay
{
// This node will stop downstream execution for a period of time before continuing.
// The period of time is in 'Duration', measured in seconds, and begins when 'Execute
// In' is activated. Once the delay period has elapsed the node activates its 'Finished'
// signal, and execution continues downstream.
// It is important to note that the execution can only resume when the application next
// updates. As the length of the application update varies based on the complexity,
// the 'Duration' is merely a minimum delay time. For example if the 'Duration' is 10
// seconds and the application update time is 7 seconds then the node will trigger somewhere
// around the 14 second mark. After the first execution only 7 seconds will have elapsed,
// less than the 10 seconds requested, and at the second execution it will be a total
// of 14 seconds, which will be recognized as satisfying the 'Duration' requirement.
class OgnDelayDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnDelay.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnDelay.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnDelay.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnDelay.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnDelay;
    static std::tuple<int, int, int>sm_generatorVersionOgnDelay;
    static std::tuple<int, int, int>sm_targetVersionOgnDelay;
    static constexpr size_t staticAttributeCount = 5;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : duration{offset}
        , execIn{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const double,ogn::kCpu> duration;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : finished{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> finished;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnDelayDatabase(NodeObj const& nodeObjParam)
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
    OgnDelayDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnDelayDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnDelayDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::duration.m_token, inputs::execIn.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::finished.m_token
                )
            , kAccordingToContextIndex);
            inputs.duration.setContext(contextObj);
            inputs.duration.setHandle(std::get<0>(inputDataHandles0));
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<1>(inputDataHandles0));
            outputs.finished.setContext(contextObj);
            outputs.finished.setHandle(std::get<0>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.Delay");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::duration.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::finished.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Delay");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,time");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "This node will stop downstream execution for a period of time before continuing. The period of time is in 'Duration', measured in seconds, and begins when 'Execute In' is activated. Once the delay period has elapsed the node activates its 'Finished' signal, and execution continues downstream.\nIt is important to note that the execution can only resume when the application next updates. As the length of the application update varies based on the complexity, the 'Duration' is merely a minimum delay time. For example if the 'Duration' is 10 seconds and the application update time is 7 seconds then the node will trigger somewhere around the 14 second mark. After the first execution only 7 seconds will have elapsed, less than the 10 seconds requested, and at the second execution it will be a total of 14 seconds, which will be recognized as satisfying the 'Duration' requirement.");
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::duration.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The duration of the delay in seconds.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Duration");
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute In");
        attr = iNode->getAttributeByToken(nodeObj, outputs::finished.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After 'Duration' has elapsed signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Finished");
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
        sm_stateManagerOgnDelay.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.duration.isValid()
            && inputs.execIn.isValid()
            && outputs.finished.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.duration.invalidateCachedPointer();
            inputs.execIn.invalidateCachedPointer();
            outputs.finished.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::duration.m_token) {
                inputs.duration.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::finished.m_token) {
                outputs.finished.invalidateCachedPointer();
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
        if( !inputs.duration.canVectorize()
            || !inputs.execIn.canVectorize()
            || !outputs.finished.canVectorize()
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
        if(token == inputs::duration.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.duration.setHandle(hdl);
            return;
        }
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == outputs::finished.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.finished.setHandle(hdl);
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
ogn::StateManager OgnDelayDatabase::sm_stateManagerOgnDelay;
std::tuple<int, int, int> OgnDelayDatabase::sm_generatorVersionOgnDelay{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnDelayDatabase::sm_targetVersionOgnDelay{std::make_tuple(2,184,5)};
}
using namespace IOgnDelay;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnDelay, OgnDelayDatabase> s_registration("omni.graph.action.Delay", 2, "omni.graph.action_nodes"); \
}
