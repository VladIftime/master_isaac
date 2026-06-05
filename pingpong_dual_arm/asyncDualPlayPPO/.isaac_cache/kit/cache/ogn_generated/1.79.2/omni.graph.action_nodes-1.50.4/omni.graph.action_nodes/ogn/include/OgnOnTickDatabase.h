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

namespace OgnOnTickAttributes
{
namespace inputs
{
using framePeriod_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> framePeriod("inputs:framePeriod", "uint", kExtendedAttributeType_Regular, 0);
using onlyPlayback_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> onlyPlayback("inputs:onlyPlayback", "bool", kExtendedAttributeType_Regular, true);
}
namespace outputs
{
using absoluteSimTime_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> absoluteSimTime("outputs:absoluteSimTime", "double", kExtendedAttributeType_Regular);
using deltaSeconds_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> deltaSeconds("outputs:deltaSeconds", "double", kExtendedAttributeType_Regular);
using frame_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> frame("outputs:frame", "double", kExtendedAttributeType_Regular);
using isPlaying_t = bool&;
ogn::AttributeInitializer<bool, ogn::kOgnOutput> isPlaying("outputs:isPlaying", "bool", kExtendedAttributeType_Regular);
using tick_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> tick("outputs:tick", "execution", kExtendedAttributeType_Regular);
using time_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> time("outputs:time", "double", kExtendedAttributeType_Regular);
using timeSinceStart_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnOutput> timeSinceStart("outputs:timeSinceStart", "double", kExtendedAttributeType_Regular);
}
namespace state
{
using accumulatedSeconds_t = double&;
ogn::AttributeInitializer<double, ogn::kOgnState> accumulatedSeconds("state:accumulatedSeconds", "double", kExtendedAttributeType_Regular, 0.0);
using frameCount_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnState> frameCount("state:frameCount", "uint", kExtendedAttributeType_Regular, 0);
}
}
using namespace OgnOnTickAttributes;
namespace IOgnOnTick
{
// Activates execution of the downstream graph at a regular multiple of the application's
// refresh rate. As the application runs the this node will skip every 'Update Period'
// executions. Typically you might set this to a larger value if the downstream graph
// is doing something you do not want to happen too frequently, such as printing out
// debugging messages or polling some input that does not get triggered very often.
// The actual refresh rate timing will depend on external factors such as rendering
// time and scene complexity. In addition to the activation signal, the outputs also
// contain global time information for convenience.
class OgnOnTickDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnTick.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnTick.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnTick.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnTick.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnTick;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnTick;
    static std::tuple<int, int, int>sm_targetVersionOgnOnTick;
    static constexpr size_t staticAttributeCount = 13;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : framePeriod{offset}
        , onlyPlayback{offset}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> framePeriod;
        ogn::SimpleInput<const bool,ogn::kCpu> onlyPlayback;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : absoluteSimTime{offset}
        , deltaSeconds{offset}
        , frame{offset}
        , isPlaying{offset}
        , tick{offset,AttributeRole::eExecution}
        , time{offset}
        , timeSinceStart{offset}
        {}
        ogn::SimpleOutput<double,ogn::kCpu> absoluteSimTime;
        ogn::SimpleOutput<double,ogn::kCpu> deltaSeconds;
        ogn::SimpleOutput<double,ogn::kCpu> frame;
        ogn::SimpleOutput<bool,ogn::kCpu> isPlaying;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> tick;
        ogn::SimpleOutput<double,ogn::kCpu> time;
        ogn::SimpleOutput<double,ogn::kCpu> timeSinceStart;
    } outputs;

    struct stateT {
        stateT(size_t const& offset)
        : accumulatedSeconds{offset}
        , frameCount{offset}
        {}
        ogn::SimpleState<double,ogn::kCpu> accumulatedSeconds;
        ogn::SimpleState<uint32_t,ogn::kCpu> frameCount;
    } state;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnTickDatabase(NodeObj const& nodeObjParam)
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
    OgnOnTickDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnTickDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnTickDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::framePeriod.m_token, inputs::onlyPlayback.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle,
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::absoluteSimTime.m_token, outputs::deltaSeconds.m_token, outputs::frame.m_token, outputs::isPlaying.m_token,
                    outputs::tick.m_token, outputs::time.m_token, outputs::timeSinceStart.m_token
                )
            , kAccordingToContextIndex);
            auto stateDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    state::accumulatedSeconds.m_token, state::frameCount.m_token
                )
            , kAccordingToContextIndex);
            inputs.framePeriod.setContext(contextObj);
            inputs.framePeriod.setHandle(std::get<0>(inputDataHandles0));
            inputs.onlyPlayback.setContext(contextObj);
            inputs.onlyPlayback.setHandle(std::get<1>(inputDataHandles0));
            outputs.absoluteSimTime.setContext(contextObj);
            outputs.absoluteSimTime.setHandle(std::get<0>(outputDataHandles0));
            outputs.deltaSeconds.setContext(contextObj);
            outputs.deltaSeconds.setHandle(std::get<1>(outputDataHandles0));
            outputs.frame.setContext(contextObj);
            outputs.frame.setHandle(std::get<2>(outputDataHandles0));
            outputs.isPlaying.setContext(contextObj);
            outputs.isPlaying.setHandle(std::get<3>(outputDataHandles0));
            outputs.tick.setContext(contextObj);
            outputs.tick.setHandle(std::get<4>(outputDataHandles0));
            outputs.time.setContext(contextObj);
            outputs.time.setHandle(std::get<5>(outputDataHandles0));
            outputs.timeSinceStart.setContext(contextObj);
            outputs.timeSinceStart.setHandle(std::get<6>(outputDataHandles0));
            state.accumulatedSeconds.setContext(contextObj);
            state.accumulatedSeconds.setHandle(std::get<0>(stateDataHandles0));
            state.frameCount.setContext(contextObj);
            state.frameCount.setHandle(std::get<1>(stateDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnTick");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::framePeriod.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::onlyPlayback.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::absoluteSimTime.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::deltaSeconds.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::frame.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::isPlaying.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::tick.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::time.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::timeSinceStart.initialize(iToken, *iNodeType, nodeTypeObj);

        state::accumulatedSeconds.initialize(iToken, *iNodeType, nodeTypeObj);
        state::frameCount.initialize(iToken, *iNodeType, nodeTypeObj);
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On Tick");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,event");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Activates execution of the downstream graph at a regular multiple of the application's refresh rate. As the application runs the this node will skip every 'Update Period' executions. Typically you might set this to a larger value if the downstream graph is doing something you do not want to happen too frequently, such as printing out debugging messages or polling some input that does not get triggered very often. The actual refresh rate timing will depend on external factors such as rendering time and scene complexity. In addition to the activation signal, the outputs also contain global time information for convenience.");
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::framePeriod.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The number of application refreshes to skip between executions of this node.\nFor example, the default 0 means no skipping so this node executes on every application update,\n1 means every other update. This can be used to rate-limit updates.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Update Period (Ticks)");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "0");
        attr = iNode->getAttributeByToken(nodeObj, inputs::onlyPlayback.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the node is only executed while the Stage is being played.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Only Simulate On Play");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "true");
        attr = iNode->getAttributeByToken(nodeObj, outputs::absoluteSimTime.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The accumulated total of elapsed times between rendered frames. This is independent of any\nexecutions skipped as a result of a non-zero 'Update Period'.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Absolute Simulation Time (Seconds)");
        attr = iNode->getAttributeByToken(nodeObj, outputs::deltaSeconds.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The accumulated graph evaluation time since the last time the graph downstream of 'Tick' was activated.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Delta (Seconds)");
        attr = iNode->getAttributeByToken(nodeObj, outputs::frame.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The global animation time in frames, equivalent to (Animation Time * FramesPerSecond), during playback.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Animation Time (Frames)");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isPlaying.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "True during global animation timeline playback.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Is Playing");
        attr = iNode->getAttributeByToken(nodeObj, outputs::tick.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "At each specified update tick, signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Tick");
        attr = iNode->getAttributeByToken(nodeObj, outputs::time.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The global animation time in seconds during playback.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Animation Time (Seconds)");
        attr = iNode->getAttributeByToken(nodeObj, outputs::timeSinceStart.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Elapsed time since the application started.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Time Since Start (Seconds)");
        attr = iNode->getAttributeByToken(nodeObj, state::accumulatedSeconds.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Accumulated time since the last activation of the downstream graph.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "0");
        attr = iNode->getAttributeByToken(nodeObj, state::frameCount.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Accumulated frames since the last activation of the downstream graph.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "0");
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
        sm_stateManagerOgnOnTick.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.framePeriod.isValid()
            && inputs.onlyPlayback.isValid()
            && outputs.absoluteSimTime.isValid()
            && outputs.deltaSeconds.isValid()
            && outputs.frame.isValid()
            && outputs.isPlaying.isValid()
            && outputs.tick.isValid()
            && outputs.time.isValid()
            && outputs.timeSinceStart.isValid()
            && state.accumulatedSeconds.isValid()
            && state.frameCount.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.framePeriod.invalidateCachedPointer();
            inputs.onlyPlayback.invalidateCachedPointer();
            outputs.absoluteSimTime.invalidateCachedPointer();
            outputs.deltaSeconds.invalidateCachedPointer();
            outputs.frame.invalidateCachedPointer();
            outputs.isPlaying.invalidateCachedPointer();
            outputs.tick.invalidateCachedPointer();
            outputs.time.invalidateCachedPointer();
            outputs.timeSinceStart.invalidateCachedPointer();
            state.accumulatedSeconds.invalidateCachedPointer();
            state.frameCount.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::framePeriod.m_token) {
                inputs.framePeriod.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::onlyPlayback.m_token) {
                inputs.onlyPlayback.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::absoluteSimTime.m_token) {
                outputs.absoluteSimTime.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::deltaSeconds.m_token) {
                outputs.deltaSeconds.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::frame.m_token) {
                outputs.frame.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::isPlaying.m_token) {
                outputs.isPlaying.invalidateCachedPointer();
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
            if(attrib == outputs::timeSinceStart.m_token) {
                outputs.timeSinceStart.invalidateCachedPointer();
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
        if( !inputs.framePeriod.canVectorize()
            || !inputs.onlyPlayback.canVectorize()
            || !outputs.absoluteSimTime.canVectorize()
            || !outputs.deltaSeconds.canVectorize()
            || !outputs.frame.canVectorize()
            || !outputs.isPlaying.canVectorize()
            || !outputs.tick.canVectorize()
            || !outputs.time.canVectorize()
            || !outputs.timeSinceStart.canVectorize()
            || !state.accumulatedSeconds.canVectorize()
            || !state.frameCount.canVectorize()
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
        if(token == inputs::framePeriod.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.framePeriod.setHandle(hdl);
            return;
        }
        if(token == inputs::onlyPlayback.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.onlyPlayback.setHandle(hdl);
            return;
        }
        if(token == outputs::absoluteSimTime.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.absoluteSimTime.setHandle(hdl);
            return;
        }
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
        if(token == outputs::isPlaying.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.isPlaying.setHandle(hdl);
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
        if(token == outputs::timeSinceStart.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.timeSinceStart.setHandle(hdl);
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
ogn::StateManager OgnOnTickDatabase::sm_stateManagerOgnOnTick;
std::tuple<int, int, int> OgnOnTickDatabase::sm_generatorVersionOgnOnTick{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnTickDatabase::sm_targetVersionOgnOnTick{std::make_tuple(2,184,5)};
}
using namespace IOgnOnTick;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnTick, OgnOnTickDatabase> s_registration("omni.graph.action.OnTick", 2, "omni.graph.action_nodes"); \
}
