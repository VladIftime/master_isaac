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
#include <array>
#include <omni/graph/core/Type.h>
#include <omni/graph/core/ogn/ArrayAttribute.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>

namespace OgnForEachTargetAttributes
{
namespace inputs
{
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using targets_t = const ogn::const_array<TargetPath>&;
ogn::AttributeInitializer<const TargetPath*, ogn::kOgnInput> targets("inputs:targets", "target", kExtendedAttributeType_Regular, nullptr, 0);
}
namespace outputs
{
using arrayIndex_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnOutput> arrayIndex("outputs:arrayIndex", "int", kExtendedAttributeType_Regular);
using finished_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> finished("outputs:finished", "execution", kExtendedAttributeType_Regular);
using loopBody_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> loopBody("outputs:loopBody", "execution", kExtendedAttributeType_Regular);
using target_t = ogn::array<TargetPath>&;
ogn::AttributeInitializer<TargetPath*, ogn::kOgnOutput> target("outputs:target", "target", kExtendedAttributeType_Regular, nullptr, 0);
}
namespace state
{
}
}
using namespace OgnForEachTargetAttributes;
namespace IOgnForEachTarget
{
// Activates the 'Loop Body' signal once for each target in 'Targets', making the current
// array member available in the output 'Target' with its index in 'Array Index'. After
// every element of 'Targets' has been processed the 'Finished' signal is activated.
// All of this will happen in a single execution of the node, giving you the ability
// to evaluate a downstream graph multiple times with different inputs coming from the
// changing 'Target' output.
class OgnForEachTargetDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnForEachTarget.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnForEachTarget.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnForEachTarget.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnForEachTarget.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnForEachTarget;
    static std::tuple<int, int, int>sm_generatorVersionOgnForEachTarget;
    static std::tuple<int, int, int>sm_targetVersionOgnForEachTarget;
    static constexpr size_t staticAttributeCount = 8;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : execIn{offset,AttributeRole::eExecution}
        , targets{offset,AttributeRole::eTarget}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::ArrayInput<const TargetPath,ogn::kCpu> targets;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : arrayIndex{offset}
        , finished{offset,AttributeRole::eExecution}
        , loopBody{offset,AttributeRole::eExecution}
        , target{offset,AttributeRole::eTarget}
        {}
        ogn::SimpleOutput<int,ogn::kCpu> arrayIndex;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> finished;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> loopBody;
        ogn::ArrayOutput<TargetPath,ogn::kCpu> target;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnForEachTargetDatabase(NodeObj const& nodeObjParam)
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
    OgnForEachTargetDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnForEachTargetDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnForEachTargetDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::execIn.m_token, inputs::targets.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::arrayIndex.m_token, outputs::finished.m_token, outputs::loopBody.m_token, outputs::target.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.targets.setContext(contextObj);
            inputs.targets.setHandle(std::get<1>(inputDataHandles0));
            outputs.arrayIndex.setContext(contextObj);
            outputs.arrayIndex.setHandle(std::get<0>(outputDataHandles0));
            outputs.finished.setContext(contextObj);
            outputs.finished.setHandle(std::get<1>(outputDataHandles0));
            outputs.loopBody.setContext(contextObj);
            outputs.loopBody.setHandle(std::get<2>(outputDataHandles0));
            outputs.target.setContext(contextObj);
            outputs.target.setHandle(std::get<3>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.ForEachTarget");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::targets.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::arrayIndex.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::finished.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::loopBody.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::target.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "For Each Target Loop");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Activates the 'Loop Body' signal once for each target in 'Targets', making the current array member available in the output 'Target' with its index in 'Array Index'. After every element of 'Targets' has been processed the 'Finished' signal is activated. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Target' output.");
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
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataIconPath, "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForEachTarget.svg");
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr = iNode->getAttributeByToken(nodeObj, inputs::targets.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The targets array to loop over");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowMultiInputs, "1");
        attr = iNode->getAttributeByToken(nodeObj, outputs::arrayIndex.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The value of the current index being visited by the loop. Keeps the value of the last index\nafter the loop has completed. The index starts at zero and increments by one as it\nwalks through the members of 'Targets'.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::finished.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the final element of 'Targets' has been processed signal the graph that\nexecution can continue downstream.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::loopBody.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "For each member of 'Targets' signal the graph that execution can continue downstream.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::target.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The current member of 'Targets' being visited by the loop. Keeps the value of the last\narray element after the loop has completed.");
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
        sm_stateManagerOgnForEachTarget.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.execIn.isValid()
            && inputs.targets.isValid()
            && outputs.arrayIndex.isValid()
            && outputs.finished.isValid()
            && outputs.loopBody.isValid()
            && outputs.target.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.execIn.invalidateCachedPointer();
            inputs.targets.invalidateCachedPointer();
            outputs.arrayIndex.invalidateCachedPointer();
            outputs.finished.invalidateCachedPointer();
            outputs.loopBody.invalidateCachedPointer();
            outputs.target.invalidateCachedPointer();
            return;
        }
        inputs.targets.invalidateCachedPointer();
        outputs.target.invalidateCachedPointer();
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::targets.m_token) {
                inputs.targets.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::arrayIndex.m_token) {
                outputs.arrayIndex.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::finished.m_token) {
                outputs.finished.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::loopBody.m_token) {
                outputs.loopBody.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::target.m_token) {
                outputs.target.invalidateCachedPointer();
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
            || !outputs.arrayIndex.canVectorize()
            || !outputs.finished.canVectorize()
            || !outputs.loopBody.canVectorize()
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
        if(token == inputs::targets.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.targets.setHandle(hdl);
            return;
        }
        if(token == outputs::arrayIndex.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.arrayIndex.setHandle(hdl);
            return;
        }
        if(token == outputs::finished.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.finished.setHandle(hdl);
            return;
        }
        if(token == outputs::loopBody.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.loopBody.setHandle(hdl);
            return;
        }
        if(token == outputs::target.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.target.setHandle(hdl);
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
ogn::StateManager OgnForEachTargetDatabase::sm_stateManagerOgnForEachTarget;
std::tuple<int, int, int> OgnForEachTargetDatabase::sm_generatorVersionOgnForEachTarget{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnForEachTargetDatabase::sm_targetVersionOgnForEachTarget{std::make_tuple(2,184,5)};
}
using namespace IOgnForEachTarget;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnForEachTarget, OgnForEachTargetDatabase> s_registration("omni.graph.action.ForEachTarget", 1, "omni.graph.action_nodes"); \
}
