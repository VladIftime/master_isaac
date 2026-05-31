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
#include <carb/tokens/TokensUtils.h>
#include <omni/graph/core/Type.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>

namespace OgnForLoopAttributes
{
namespace inputs
{
using breakLoop_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> breakLoop("inputs:breakLoop", "execution", kExtendedAttributeType_Regular);
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using start_t = const int&;
ogn::AttributeInitializer<const int, ogn::kOgnInput> start("inputs:start", "int", kExtendedAttributeType_Regular, 0);
using step_t = const int&;
ogn::AttributeInitializer<const int, ogn::kOgnInput> step("inputs:step", "int", kExtendedAttributeType_Regular, 1);
using stop_t = const int&;
ogn::AttributeInitializer<const int, ogn::kOgnInput> stop("inputs:stop", "int", kExtendedAttributeType_Regular, 0);
}
namespace outputs
{
using finished_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> finished("outputs:finished", "execution", kExtendedAttributeType_Regular);
using index_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnOutput> index("outputs:index", "int", kExtendedAttributeType_Regular);
using loopBody_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> loopBody("outputs:loopBody", "execution", kExtendedAttributeType_Regular);
using value_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnOutput> value("outputs:value", "int", kExtendedAttributeType_Regular);
}
namespace state
{
using i_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnState> i("state:i", "int", kExtendedAttributeType_Regular, -1);
}
}
using namespace OgnForLoopAttributes;
namespace IOgnForLoop
{
// Executes the a loop body once for each value within a range. When step is positive,
// the values in the range are determined by the formula:
// r[i] = start + step*i, i >= 0 & r[i] < stop.
// When step is negative the constraint  is instead r[i] > stop. A step of zero is an
// error.
// The break input can be used to break out of the loop before the last index.  The
// finished output is executed after all iterations are complete, or when the loop was
// broken. All of this will happen in a single execution of the node, giving you the
// ability to evaluate a downstream graph multiple times with different inputs coming
// from the changing 'Value' and 'Index' outputs.
class OgnForLoopDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnForLoop.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnForLoop.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnForLoop.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnForLoop.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnForLoop;
    static std::tuple<int, int, int>sm_generatorVersionOgnForLoop;
    static std::tuple<int, int, int>sm_targetVersionOgnForLoop;
    static constexpr size_t staticAttributeCount = 12;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : breakLoop{offset,AttributeRole::eExecution}
        , execIn{offset,AttributeRole::eExecution}
        , start{offset}
        , step{offset}
        , stop{offset}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> breakLoop;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::SimpleInput<const int,ogn::kCpu> start;
        ogn::SimpleInput<const int,ogn::kCpu> step;
        ogn::SimpleInput<const int,ogn::kCpu> stop;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : finished{offset,AttributeRole::eExecution}
        , index{offset}
        , loopBody{offset,AttributeRole::eExecution}
        , value{offset}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> finished;
        ogn::SimpleOutput<int,ogn::kCpu> index;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> loopBody;
        ogn::SimpleOutput<int,ogn::kCpu> value;
    } outputs;

    struct stateT {
        stateT(size_t const& offset)
        : i{offset}
        {}
        ogn::SimpleState<int,ogn::kCpu> i;
    } state;

    //Only use this constructor for temporary stack-allocated object:
    OgnForLoopDatabase(NodeObj const& nodeObjParam)
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
    OgnForLoopDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnForLoopDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnForLoopDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle,
                ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::breakLoop.m_token, inputs::execIn.m_token, inputs::start.m_token, inputs::step.m_token,
                    inputs::stop.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::finished.m_token, outputs::index.m_token, outputs::loopBody.m_token, outputs::value.m_token
                )
            , kAccordingToContextIndex);
            auto stateDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    state::i.m_token
                )
            , kAccordingToContextIndex);
            inputs.breakLoop.setContext(contextObj);
            inputs.breakLoop.setHandle(std::get<0>(inputDataHandles0));
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<1>(inputDataHandles0));
            inputs.start.setContext(contextObj);
            inputs.start.setHandle(std::get<2>(inputDataHandles0));
            inputs.step.setContext(contextObj);
            inputs.step.setHandle(std::get<3>(inputDataHandles0));
            inputs.stop.setContext(contextObj);
            inputs.stop.setHandle(std::get<4>(inputDataHandles0));
            outputs.finished.setContext(contextObj);
            outputs.finished.setHandle(std::get<0>(outputDataHandles0));
            outputs.index.setContext(contextObj);
            outputs.index.setHandle(std::get<1>(outputDataHandles0));
            outputs.loopBody.setContext(contextObj);
            outputs.loopBody.setHandle(std::get<2>(outputDataHandles0));
            outputs.value.setContext(contextObj);
            outputs.value.setHandle(std::get<3>(outputDataHandles0));
            state.i.setContext(contextObj);
            state.i.setHandle(std::get<0>(stateDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.ForLoop");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::breakLoop.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::start.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::step.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::stop.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::finished.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::index.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::loopBody.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::value.initialize(iToken, *iNodeType, nodeTypeObj);

        state::i.initialize(iToken, *iNodeType, nodeTypeObj);
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataTags, "range");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "For Loop");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Executes the a loop body once for each value within a range. When step is positive, the values in the range are determined by the formula:\nr[i] = start + step*i, i >= 0 & r[i] < stop.\nWhen step is negative the constraint  is instead r[i] > stop. A step of zero is an error.\nThe break input can be used to break out of the loop before the last index.  The finished output is executed after all iterations are complete, or when the loop was broken. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Value' and 'Index' outputs.");
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
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataIconPath, "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForLoop.svg");
        iNodeType->setHasState(nodeTypeObj, true);
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::breakLoop.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that execution of the loop body is to be aborted. It behaves exactly as\nthe 'break' statements in Python or C++ behave. After the loop is broken the current 'Value'\nand 'Index' retain their current values and the 'Finished' signal is activated.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Break");
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "In");
        attr = iNode->getAttributeByToken(nodeObj, inputs::start.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The first value in the range to loop over.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Start");
        attr = iNode->getAttributeByToken(nodeObj, inputs::step.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The step size of the range. This number is added to the 'Value' after each execution.\nThe value can be negative to step backwards, however it is an error if the value is zero.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Step");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "1");
        attr = iNode->getAttributeByToken(nodeObj, inputs::stop.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The limiting value of the range to loop over. It may or may not be equal to the final\nvalue of the loop, depending on the 'Step' value. For example if 'Start' is 1, 'Step' is 2,\nand 'Stop' is 4 then the loop will run with output 'Value's equal to 1 and 3, but not 5.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Stop");
        attr = iNode->getAttributeByToken(nodeObj, outputs::finished.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When either the 'Value' has reached or exceeded 'Stop', or 'Signal Break' has activated\nsignal the graph that execution can continue downstream.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::index.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The current value of the index when the loop is active, or the value the index had when the\n'Stop' threshold was met or exceeded after the loop has completed. The 'Index' value starts\nat zero after the first execution and increments by one each time the loop body runs.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::loopBody.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "For each execution where the 'Value' is still in range signal the graph that\nexecution can continue downstream.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::value.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The current value of the range when the loop is active, or the value that met or exceeded the\n'Stop' threshold after the loop has completed.");
        attr = iNode->getAttributeByToken(nodeObj, state::i.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The next index in the range, or -1 when loop is not active.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "-1");
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
        sm_stateManagerOgnForLoop.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.breakLoop.isValid()
            && inputs.execIn.isValid()
            && inputs.start.isValid()
            && inputs.step.isValid()
            && inputs.stop.isValid()
            && outputs.finished.isValid()
            && outputs.index.isValid()
            && outputs.loopBody.isValid()
            && outputs.value.isValid()
            && state.i.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.breakLoop.invalidateCachedPointer();
            inputs.execIn.invalidateCachedPointer();
            inputs.start.invalidateCachedPointer();
            inputs.step.invalidateCachedPointer();
            inputs.stop.invalidateCachedPointer();
            outputs.finished.invalidateCachedPointer();
            outputs.index.invalidateCachedPointer();
            outputs.loopBody.invalidateCachedPointer();
            outputs.value.invalidateCachedPointer();
            state.i.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::breakLoop.m_token) {
                inputs.breakLoop.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::start.m_token) {
                inputs.start.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::step.m_token) {
                inputs.step.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::stop.m_token) {
                inputs.stop.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::finished.m_token) {
                outputs.finished.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::index.m_token) {
                outputs.index.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::loopBody.m_token) {
                outputs.loopBody.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::value.m_token) {
                outputs.value.invalidateCachedPointer();
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
        if( !inputs.breakLoop.canVectorize()
            || !inputs.execIn.canVectorize()
            || !inputs.start.canVectorize()
            || !inputs.step.canVectorize()
            || !inputs.stop.canVectorize()
            || !outputs.finished.canVectorize()
            || !outputs.index.canVectorize()
            || !outputs.loopBody.canVectorize()
            || !outputs.value.canVectorize()
            || !state.i.canVectorize()
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
        if(token == inputs::breakLoop.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.breakLoop.setHandle(hdl);
            return;
        }
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == inputs::start.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.start.setHandle(hdl);
            return;
        }
        if(token == inputs::step.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.step.setHandle(hdl);
            return;
        }
        if(token == inputs::stop.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.stop.setHandle(hdl);
            return;
        }
        if(token == outputs::finished.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.finished.setHandle(hdl);
            return;
        }
        if(token == outputs::index.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.index.setHandle(hdl);
            return;
        }
        if(token == outputs::loopBody.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.loopBody.setHandle(hdl);
            return;
        }
        if(token == outputs::value.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.value.setHandle(hdl);
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
ogn::StateManager OgnForLoopDatabase::sm_stateManagerOgnForLoop;
std::tuple<int, int, int> OgnForLoopDatabase::sm_generatorVersionOgnForLoop{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnForLoopDatabase::sm_targetVersionOgnForLoop{std::make_tuple(2,184,5)};
}
using namespace IOgnForLoop;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnForLoop, OgnForLoopDatabase> s_registration("omni.graph.action.ForLoop", 2, "omni.graph.action_nodes"); \
}
