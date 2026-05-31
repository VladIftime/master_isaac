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

namespace OgnOnPlaybackTickAttributes
{
namespace inputs
{
}
namespace outputs
{
using deltaSeconds_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> deltaSeconds("outputs:deltaSeconds", "double", kExtendedAttributeType_Regular);
using frame_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> frame("outputs:frame", "double", kExtendedAttributeType_Regular);
using tick_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> tick("outputs:tick", "execution", kExtendedAttributeType_Regular);
using time_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> time("outputs:time", "double", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnOnPlaybackTickAttributes;
namespace IOgnOnPlaybackTick
{
// For each frame tick during playback, activate the downstream graph execution. In
// addition to the activation signal, the outputs also contain the playback time values,
// taken directly from the execution context.
class OgnOnPlaybackTickDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnPlaybackTick.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnPlaybackTick.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnPlaybackTick.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnPlaybackTick.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnPlaybackTick;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnPlaybackTick;
    static std::tuple<int, int, int>sm_targetVersionOgnOnPlaybackTick;
    static constexpr size_t staticAttributeCount = 6;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct outputsT {
        outputsT(size_t const& offset)
        : deltaSeconds{offset}
        , frame{offset}
        , tick{offset,AttributeRole::eExecution}
        , time{offset}
        {}
        ogn::SimpleOutput<double,ogn::kCpu> deltaSeconds;
        ogn::SimpleOutput<double,ogn::kCpu> frame;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> tick;
        ogn::SimpleOutput<double,ogn::kCpu> time;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnPlaybackTickDatabase(NodeObj const& nodeObjParam)
    : OmniGraphDatabase()
    , outputs{m_offset.index}
    {
        GraphContextObj const* contexts = nullptr;
        NodeObj const* nodes = nullptr;
        size_t handleCount = nodeObjParam.iNode->getAutoInstances(nodeObjParam, contexts, nodes);
        _ctor(contexts, nodes, handleCount);
        _init();
    }

    CARB_DEPRECATED("Passing the graph context to the temporary stack allocated database is not necessary anymore: you can safely remove this parameter")
    OgnOnPlaybackTickDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnPlaybackTickDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnPlaybackTickDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
    : OmniGraphDatabase()
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
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::deltaSeconds.m_token, outputs::frame.m_token, outputs::tick.m_token, outputs::time.m_token
                )
            , kAccordingToContextIndex);
            outputs.deltaSeconds.setContext(contextObj);
            outputs.deltaSeconds.setHandle(std::get<0>(outputDataHandles0));
            outputs.frame.setContext(contextObj);
            outputs.frame.setHandle(std::get<1>(outputDataHandles0));
            outputs.tick.setContext(contextObj);
            outputs.tick.setHandle(std::get<2>(outputDataHandles0));
            outputs.time.setContext(contextObj);
            outputs.time.setHandle(std::get<3>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnPlaybackTick");
            return;
        }
        auto& iToken{ *iTokenPtr };


        outputs::deltaSeconds.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::frame.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::tick.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::time.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On Playback Tick");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,event");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "For each frame tick during playback, activate the downstream graph execution. In addition to the activation signal, the outputs also contain the playback time values, taken directly from the execution context.");
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
        attr = iNode->getAttributeByToken(nodeObj, outputs::deltaSeconds.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The number of seconds that have elapsed since the last update.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Delta Seconds");
        attr = iNode->getAttributeByToken(nodeObj, outputs::frame.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The global playback time in frames, equivalent to (Time * PlaybackFramesPerSecond).");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Frame");
        attr = iNode->getAttributeByToken(nodeObj, outputs::tick.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Tick");
        attr = iNode->getAttributeByToken(nodeObj, outputs::time.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The global playback time in seconds.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Time");
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
        sm_stateManagerOgnOnPlaybackTick.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && outputs.deltaSeconds.isValid()
            && outputs.frame.isValid()
            && outputs.tick.isValid()
            && outputs.time.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            outputs.deltaSeconds.invalidateCachedPointer();
            outputs.frame.invalidateCachedPointer();
            outputs.tick.invalidateCachedPointer();
            outputs.time.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == outputs::deltaSeconds.m_token) {
                outputs.deltaSeconds.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::frame.m_token) {
                outputs.frame.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::tick.m_token) {
                outputs.tick.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::time.m_token) {
                outputs.time.invalidateCachedPointer();
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
        if( !outputs.deltaSeconds.canVectorize()
            || !outputs.frame.canVectorize()
            || !outputs.tick.canVectorize()
            || !outputs.time.canVectorize()
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
        if(token == outputs::deltaSeconds.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.deltaSeconds.setHandle(hdl);
            return;
        }
        if(token == outputs::frame.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.frame.setHandle(hdl);
            return;
        }
        if(token == outputs::tick.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.tick.setHandle(hdl);
            return;
        }
        if(token == outputs::time.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.time.setHandle(hdl);
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
ogn::StateManager OgnOnPlaybackTickDatabase::sm_stateManagerOgnOnPlaybackTick;
std::tuple<int, int, int> OgnOnPlaybackTickDatabase::sm_generatorVersionOgnOnPlaybackTick{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnPlaybackTickDatabase::sm_targetVersionOgnOnPlaybackTick{std::make_tuple(2,184,5)};
}
using namespace IOgnOnPlaybackTick;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnPlaybackTick, OgnOnPlaybackTickDatabase> s_registration("omni.graph.action.OnPlaybackTick", 2, "omni.graph.action_nodes"); \
}
