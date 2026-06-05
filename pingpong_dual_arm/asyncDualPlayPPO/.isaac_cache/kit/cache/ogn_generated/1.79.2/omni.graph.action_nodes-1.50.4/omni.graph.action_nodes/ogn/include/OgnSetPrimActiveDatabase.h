#pragma once

#include <omni/graph/core/ISchedulingHints2.h>
#include <omni/graph/core/IInternal.h>
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
#include <array>
#include <omni/graph/core/Type.h>
#include <omni/graph/core/ogn/ArrayAttribute.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>

namespace OgnSetPrimActiveAttributes
{
namespace inputs
{
using active_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> active("inputs:active", "bool", kExtendedAttributeType_Regular, false);
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using prim_t = const char*&;
ogn::AttributeInitializer<const char*, ogn::kOgnInput> prim("inputs:prim", "path", kExtendedAttributeType_Regular, "", 0);
using primTarget_t = const ogn::const_array<TargetPath>&;
ogn::AttributeInitializer<const TargetPath*, ogn::kOgnInput> primTarget("inputs:primTarget", "target", kExtendedAttributeType_Regular, nullptr, 0);
}
namespace outputs
{
using execOut_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> execOut("outputs:execOut", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnSetPrimActiveAttributes;
namespace IOgnSetPrimActive
{
// Set whether a prim on the Stage is active (selected) or not. Only one prim can be
// connected to the 'Prim' input for execution. If multiple targets are present then
// a warning will be logged. The 'Active' input value will be the prim's new active
// state. If the prim cannot be found then an error will be logged. Avoiding use of
// the deprecated 'Prim Path' input will ensure that error cannot be encountered.
class OgnSetPrimActiveDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnSetPrimActive.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnSetPrimActive.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnSetPrimActive.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnSetPrimActive.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnSetPrimActive;
    static std::tuple<int, int, int>sm_generatorVersionOgnSetPrimActive;
    static std::tuple<int, int, int>sm_targetVersionOgnSetPrimActive;
    static constexpr size_t staticAttributeCount = 7;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : active{offset}
        , execIn{offset,AttributeRole::eExecution}
        , prim{offset,AttributeRole::ePath}
        , primTarget{offset,AttributeRole::eTarget}
        {}
        ogn::SimpleInput<const bool,ogn::kCpu> active;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::ArrayInput<const char,ogn::kCpu> prim;
        ogn::ArrayInput<const TargetPath,ogn::kCpu> primTarget;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : execOut{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> execOut;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnSetPrimActiveDatabase(NodeObj const& nodeObjParam)
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
    OgnSetPrimActiveDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnSetPrimActiveDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnSetPrimActiveDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::active.m_token, inputs::execIn.m_token, inputs::prim.m_token, inputs::primTarget.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::execOut.m_token
                )
            , kAccordingToContextIndex);
            inputs.active.setContext(contextObj);
            inputs.active.setHandle(std::get<0>(inputDataHandles0));
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<1>(inputDataHandles0));
            inputs.prim.setContext(contextObj);
            inputs.prim.setHandle(std::get<2>(inputDataHandles0));
            inputs.primTarget.setContext(contextObj);
            inputs.primTarget.setHandle(std::get<3>(inputDataHandles0));
            outputs.execOut.setContext(contextObj);
            outputs.execOut.setHandle(std::get<0>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.SetPrimActive");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::active.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::prim.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::primTarget.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::execOut.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Set Prim Active");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,sceneGraph");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Set whether a prim on the Stage is active (selected) or not. Only one prim can be connected to the 'Prim' input for execution. If multiple targets are present then a warning will be logged. The 'Active' input value will be the prim's new active state. If the prim cannot be found then an error will be logged. Avoiding use of the deprecated 'Prim Path' input will ensure that error cannot be encountered.");
        auto __schedulingInfo = nodeTypeObj.iNodeType->getSchedulingHints(nodeTypeObj);
        CARB_ASSERT(__schedulingInfo, "Could not acquire the scheduling hints");
        if (__schedulingInfo)
        {
            __schedulingInfo->setDataAccess(eAccessLocation::eUsd, eAccessType::eWrite);
            auto __schedulingInfo2 = omni::core::cast<ISchedulingHints2>(__schedulingInfo).get();
            if (__schedulingInfo2)
            {
            }
        }
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        const IInternal* iInternal = carb::getCachedInterface<omni::graph::core::IInternal>();
        if( ! iInternal ) {
            CARB_LOG_ERROR("IInternal not found when initializing omni.graph.action.SetPrimActive");
            return;
        }
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::active.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Whether to set the prim active or not");
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr = iNode->getAttributeByToken(nodeObj, inputs::prim.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The prim to be (de)activated");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Prim Path");
        iInternal->deprecateAttribute(attr, "Use the primTarget input instead");
        attr = iNode->getAttributeByToken(nodeObj, inputs::primTarget.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The prim to be (de)activated");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Prim");
        attr = iNode->getAttributeByToken(nodeObj, outputs::execOut.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Execute Out");
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
        sm_stateManagerOgnSetPrimActive.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.active.isValid()
            && inputs.execIn.isValid()
            && inputs.prim.isValid()
            && inputs.primTarget.isValid()
            && outputs.execOut.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.active.invalidateCachedPointer();
            inputs.execIn.invalidateCachedPointer();
            inputs.prim.invalidateCachedPointer();
            inputs.primTarget.invalidateCachedPointer();
            outputs.execOut.invalidateCachedPointer();
            return;
        }
        inputs.prim.invalidateCachedPointer();
        inputs.primTarget.invalidateCachedPointer();
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::active.m_token) {
                inputs.active.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::prim.m_token) {
                inputs.prim.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::primTarget.m_token) {
                inputs.primTarget.invalidateCachedPointer();
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
        if( !inputs.active.canVectorize()
            || !inputs.execIn.canVectorize()
            || !outputs.execOut.canVectorize()
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
        if(token == inputs::active.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.active.setHandle(hdl);
            return;
        }
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == inputs::prim.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.prim.setHandle(hdl);
            return;
        }
        if(token == inputs::primTarget.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.primTarget.setHandle(hdl);
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
ogn::StateManager OgnSetPrimActiveDatabase::sm_stateManagerOgnSetPrimActive;
std::tuple<int, int, int> OgnSetPrimActiveDatabase::sm_generatorVersionOgnSetPrimActive{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnSetPrimActiveDatabase::sm_targetVersionOgnSetPrimActive{std::make_tuple(2,184,5)};
}
using namespace IOgnSetPrimActive;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnSetPrimActive, OgnSetPrimActiveDatabase> s_registration("omni.graph.action.SetPrimActive", 2, "omni.graph.action_nodes"); \
}
