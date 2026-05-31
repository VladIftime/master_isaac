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

namespace OgnOnObjectChangeAttributes
{
namespace inputs
{
using name_t = const NameToken&;
ogn::AttributeInitializer<const NameToken, ogn::kOgnInput> name("inputs:name", "token", kExtendedAttributeType_Regular);
using onlyPlayback_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> onlyPlayback("inputs:onlyPlayback", "bool", kExtendedAttributeType_Regular, true);
using path_t = const char*&;
ogn::AttributeInitializer<const char*, ogn::kOgnInput> path("inputs:path", "path", kExtendedAttributeType_Regular);
using prim_t = const ogn::const_array<TargetPath>&;
ogn::AttributeInitializer<const TargetPath*, ogn::kOgnInput> prim("inputs:prim", "target", kExtendedAttributeType_Regular, nullptr, 0);
}
namespace outputs
{
using changed_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> changed("outputs:changed", "execution", kExtendedAttributeType_Regular);
using propertyName_t = NameToken&;
ogn::AttributeInitializer<NameToken, ogn::kOgnOutput> propertyName("outputs:propertyName", "token", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnOnObjectChangeAttributes;
namespace IOgnOnObjectChange
{
// Monitors a specific 'Property Name' on a connected 'Prim' target. When a change in
// the underlying USD is detected, activates execution of the downstream graph.
class OgnOnObjectChangeDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnObjectChange.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnObjectChange.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnObjectChange.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnObjectChange.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnObjectChange;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnObjectChange;
    static std::tuple<int, int, int>sm_targetVersionOgnOnObjectChange;
    static constexpr size_t staticAttributeCount = 8;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : name{offset}
        , onlyPlayback{offset}
        , path{offset,AttributeRole::ePath}
        , prim{offset,AttributeRole::eTarget}
        {}
        ogn::SimpleInput<const NameToken,ogn::kCpu> name;
        bool has_name() const { return name.isValid(); };
        ogn::SimpleInput<const bool,ogn::kCpu> onlyPlayback;
        ogn::ArrayInput<const char,ogn::kCpu> path;
        bool has_path() const { return path.isValid(); };
        ogn::ArrayInput<const TargetPath,ogn::kCpu> prim;
        bool has_prim() const { return prim.isValid(); };
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : changed{offset,AttributeRole::eExecution}
        , propertyName{offset}
        {}
        ogn::SimpleOutput<uint32_t,ogn::kCpu> changed;
        ogn::SimpleOutput<NameToken,ogn::kCpu> propertyName;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnObjectChangeDatabase(NodeObj const& nodeObjParam)
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
    OgnOnObjectChangeDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnObjectChangeDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnObjectChangeDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::name.m_token, inputs::onlyPlayback.m_token, inputs::path.m_token, inputs::prim.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::changed.m_token, outputs::propertyName.m_token
                )
            , kAccordingToContextIndex);
            inputs.name.setContext(contextObj);
            inputs.name.setHandle(std::get<0>(inputDataHandles0));
            inputs.onlyPlayback.setContext(contextObj);
            inputs.onlyPlayback.setHandle(std::get<1>(inputDataHandles0));
            inputs.path.setContext(contextObj);
            inputs.path.setHandle(std::get<2>(inputDataHandles0));
            inputs.prim.setContext(contextObj);
            inputs.prim.setHandle(std::get<3>(inputDataHandles0));
            outputs.changed.setContext(contextObj);
            outputs.changed.setHandle(std::get<0>(outputDataHandles0));
            outputs.propertyName.setContext(contextObj);
            outputs.propertyName.setHandle(std::get<1>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnObjectChange");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::name.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::onlyPlayback.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::path.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::prim.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::changed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::propertyName.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On USD Object Change");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,event");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Monitors a specific 'Property Name' on a connected 'Prim' target. When a change in the underlying USD is detected, activates execution of the downstream graph.");
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
        const IInternal* iInternal = carb::getCachedInterface<omni::graph::core::IInternal>();
        if( ! iInternal ) {
            CARB_LOG_ERROR("IInternal not found when initializing omni.graph.action.OnObjectChange");
            return;
        }
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::name.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The name of the property of interest on the USD prim being monitored.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Property Name");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setIsOptionalForCompute(attr, true);
        attr = iNode->getAttributeByToken(nodeObj, inputs::onlyPlayback.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the node is only executed while the Stage is being played.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Only Simulate On Play");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "true");
        attr = iNode->getAttributeByToken(nodeObj, inputs::path.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The path of object of interest (property or prim). If the prim input has a target, this is ignored");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Path");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setIsOptionalForCompute(attr, true);
        iInternal->deprecateAttribute(attr, "Use prim input instead");
        attr = iNode->getAttributeByToken(nodeObj, inputs::prim.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The USD prim being monitored.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Prim");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setIsOptionalForCompute(attr, true);
        attr = iNode->getAttributeByToken(nodeObj, outputs::changed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the watched property changes signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Changed");
        attr = iNode->getAttributeByToken(nodeObj, outputs::propertyName.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The name of the property on which the change was detected.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Property Name");
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
        sm_stateManagerOgnOnObjectChange.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.onlyPlayback.isValid()
            && outputs.changed.isValid()
            && outputs.propertyName.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.name.invalidateCachedPointer();
            inputs.onlyPlayback.invalidateCachedPointer();
            inputs.path.invalidateCachedPointer();
            inputs.prim.invalidateCachedPointer();
            outputs.changed.invalidateCachedPointer();
            outputs.propertyName.invalidateCachedPointer();
            return;
        }
        inputs.path.invalidateCachedPointer();
        inputs.prim.invalidateCachedPointer();
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::name.m_token) {
                inputs.name.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::onlyPlayback.m_token) {
                inputs.onlyPlayback.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::path.m_token) {
                inputs.path.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::prim.m_token) {
                inputs.prim.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::changed.m_token) {
                outputs.changed.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::propertyName.m_token) {
                outputs.propertyName.invalidateCachedPointer();
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
        if( !inputs.name.canVectorize()
            || !inputs.onlyPlayback.canVectorize()
            || !outputs.changed.canVectorize()
            || !outputs.propertyName.canVectorize()
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
        if(token == inputs::name.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.name.setHandle(hdl);
            return;
        }
        if(token == inputs::onlyPlayback.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.onlyPlayback.setHandle(hdl);
            return;
        }
        if(token == inputs::path.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.path.setHandle(hdl);
            return;
        }
        if(token == inputs::prim.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.prim.setHandle(hdl);
            return;
        }
        if(token == outputs::changed.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.changed.setHandle(hdl);
            return;
        }
        if(token == outputs::propertyName.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.propertyName.setHandle(hdl);
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
ogn::StateManager OgnOnObjectChangeDatabase::sm_stateManagerOgnOnObjectChange;
std::tuple<int, int, int> OgnOnObjectChangeDatabase::sm_generatorVersionOgnOnObjectChange{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnObjectChangeDatabase::sm_targetVersionOgnOnObjectChange{std::make_tuple(2,184,5)};
}
using namespace IOgnOnObjectChange;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnObjectChange, OgnOnObjectChangeDatabase> s_registration("omni.graph.action.OnObjectChange", 5, "omni.graph.action_nodes"); \
}
