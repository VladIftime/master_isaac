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
#include <omni/graph/core/ogn/ArrayAttribute.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>

namespace OgnAddPrimRelationshipAttributes
{
namespace inputs
{
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
using name_t = const NameToken&;
ogn::AttributeInitializer<const NameToken, ogn::kOgnInput> name("inputs:name", "token", kExtendedAttributeType_Regular);
using path_t = const char*&;
ogn::AttributeInitializer<const char*, ogn::kOgnInput> path("inputs:path", "path", kExtendedAttributeType_Regular, "", 0);
using target_t = const char*&;
ogn::AttributeInitializer<const char*, ogn::kOgnInput> target("inputs:target", "path", kExtendedAttributeType_Regular, "", 0);
}
namespace outputs
{
using isSuccessful_t = bool&;
ogn::AttributeInitializer<bool, ogn::kOgnOutput> isSuccessful("outputs:isSuccessful", "bool", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnAddPrimRelationshipAttributes;
namespace IOgnAddPrimRelationship
{
// Adds a target path to a relationship property. If the relationship property does
// not exist it will be created. Duplicate targets will not be added.
class OgnAddPrimRelationshipDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnAddPrimRelationship.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnAddPrimRelationship.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnAddPrimRelationship.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnAddPrimRelationship.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnAddPrimRelationship;
    static std::tuple<int, int, int>sm_generatorVersionOgnAddPrimRelationship;
    static std::tuple<int, int, int>sm_targetVersionOgnAddPrimRelationship;
    static constexpr size_t staticAttributeCount = 7;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : execIn{offset,AttributeRole::eExecution}
        , name{offset}
        , path{offset,AttributeRole::ePath}
        , target{offset,AttributeRole::ePath}
        {}
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
        ogn::SimpleInput<const NameToken,ogn::kCpu> name;
        ogn::ArrayInput<const char,ogn::kCpu> path;
        ogn::ArrayInput<const char,ogn::kCpu> target;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : isSuccessful{offset}
        {}
        ogn::SimpleOutput<bool,ogn::kCpu> isSuccessful;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnAddPrimRelationshipDatabase(NodeObj const& nodeObjParam)
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
    OgnAddPrimRelationshipDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnAddPrimRelationshipDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnAddPrimRelationshipDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::execIn.m_token, inputs::name.m_token, inputs::path.m_token, inputs::target.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::isSuccessful.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.name.setContext(contextObj);
            inputs.name.setHandle(std::get<1>(inputDataHandles0));
            inputs.path.setContext(contextObj);
            inputs.path.setHandle(std::get<2>(inputDataHandles0));
            inputs.target.setContext(contextObj);
            inputs.target.setHandle(std::get<3>(inputDataHandles0));
            outputs.isSuccessful.setContext(contextObj);
            outputs.isSuccessful.setHandle(std::get<0>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.AddPrimRelationship");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::name.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::path.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::target.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::isSuccessful.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "Add Prim Relationship");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "sceneGraph");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Adds a target path to a relationship property. If the relationship property does not exist it will be created. Duplicate targets will not be added.");
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
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr = iNode->getAttributeByToken(nodeObj, inputs::name.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Name of the relationship property to be modified or added.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Relationship Name");
        attr = iNode->getAttributeByToken(nodeObj, inputs::path.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Path of the prim with the relationship property.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Prim Path");
        attr = iNode->getAttributeByToken(nodeObj, inputs::target.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The target path to be added, which may be a prim, attribute or another relationship.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Target Path");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isSuccessful.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Whether the node has successfully added the new target to the relationship.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Is Successful");
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
        sm_stateManagerOgnAddPrimRelationship.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.execIn.isValid()
            && inputs.name.isValid()
            && inputs.path.isValid()
            && inputs.target.isValid()
            && outputs.isSuccessful.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.execIn.invalidateCachedPointer();
            inputs.name.invalidateCachedPointer();
            inputs.path.invalidateCachedPointer();
            inputs.target.invalidateCachedPointer();
            outputs.isSuccessful.invalidateCachedPointer();
            return;
        }
        inputs.path.invalidateCachedPointer();
        inputs.target.invalidateCachedPointer();
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::name.m_token) {
                inputs.name.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::path.m_token) {
                inputs.path.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::target.m_token) {
                inputs.target.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::isSuccessful.m_token) {
                outputs.isSuccessful.invalidateCachedPointer();
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
            || !inputs.name.canVectorize()
            || !outputs.isSuccessful.canVectorize()
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
        if(token == inputs::name.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.name.setHandle(hdl);
            return;
        }
        if(token == inputs::path.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.path.setHandle(hdl);
            return;
        }
        if(token == inputs::target.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.target.setHandle(hdl);
            return;
        }
        if(token == outputs::isSuccessful.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.isSuccessful.setHandle(hdl);
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
ogn::StateManager OgnAddPrimRelationshipDatabase::sm_stateManagerOgnAddPrimRelationship;
std::tuple<int, int, int> OgnAddPrimRelationshipDatabase::sm_generatorVersionOgnAddPrimRelationship{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnAddPrimRelationshipDatabase::sm_targetVersionOgnAddPrimRelationship{std::make_tuple(2,184,5)};
}
using namespace IOgnAddPrimRelationship;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnAddPrimRelationship, OgnAddPrimRelationshipDatabase> s_registration("omni.graph.action.AddPrimRelationship", 1, "omni.graph.action_nodes"); \
}
