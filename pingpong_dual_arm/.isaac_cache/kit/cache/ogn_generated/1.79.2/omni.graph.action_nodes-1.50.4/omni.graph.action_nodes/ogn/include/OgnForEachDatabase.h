#pragma once

#include <omni/graph/core/ogn/UsdTypes.h>
using namespace pxr;

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
#include <omni/graph/core/ogn/SimpleRuntimeAttribute.h>

namespace OgnForEachAttributes
{
namespace inputs
{
using arrayIn_t = const ogn::RuntimeAttribute<ogn::kOgnInput, ogn::kCpu>&;
ogn::AttributeInitializer<const ogn::RuntimeAttribute<ogn::kOgnInput, ogn::kCpu>, ogn::kOgnInput> arrayIn("inputs:arrayIn", "bool[],colord[3][],colord[4][],colorf[3][],colorf[4][],colorh[3][],colorh[4][],double[2][],double[3][],double[4][],double[],float[2][],float[3][],float[4][],float[],frame[4][],half[2][],half[3][],half[4][],half[],int64[],int[2][],int[3][],int[4][],int[],matrixd[2][],matrixd[3][],matrixd[4][],normald[3][],normalf[3][],normalh[3][],pointd[3][],pointf[3][],pointh[3][],quatd[4][],quatf[4][],quath[4][],texcoordd[2][],texcoordd[3][],texcoordf[2][],texcoordf[3][],texcoordh[2][],texcoordh[3][],timecode[],token[],transform[4][],uchar[],uint64[],uint[],vectord[3][],vectorf[3][],vectorh[3][]", kExtendedAttributeType_Union);
using execIn_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> execIn("inputs:execIn", "execution", kExtendedAttributeType_Regular);
}
namespace outputs
{
using arrayIndex_t = int&;
ogn::AttributeInitializer<int, ogn::kOgnOutput> arrayIndex("outputs:arrayIndex", "int", kExtendedAttributeType_Regular);
using element_t = ogn::RuntimeAttribute<ogn::kOgnOutput, ogn::kCpu>&;
ogn::AttributeInitializer<ogn::RuntimeAttribute<ogn::kOgnOutput, ogn::kCpu>, ogn::kOgnOutput> element("outputs:element", "bool,colord[3],colord[4],colorf[3],colorf[4],colorh[3],colorh[4],double,double[2],double[3],double[4],float,float[2],float[3],float[4],frame[4],half,half[2],half[3],half[4],int,int64,int[2],int[3],int[4],matrixd[2],matrixd[3],matrixd[4],normald[3],normalf[3],normalh[3],pointd[3],pointf[3],pointh[3],quatd[4],quatf[4],quath[4],texcoordd[2],texcoordd[3],texcoordf[2],texcoordf[3],texcoordh[2],texcoordh[3],timecode,token,transform[4],uchar,uint,uint64,vectord[3],vectorf[3],vectorh[3]", kExtendedAttributeType_Union);
using finished_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> finished("outputs:finished", "execution", kExtendedAttributeType_Regular);
using loopBody_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> loopBody("outputs:loopBody", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnForEachAttributes;
namespace IOgnForEach
{
// Activates the 'Loop Body' signal once for each element in the 'Input Array', making
// the current array member available in 'Element' with its index in 'Array Index'.
// After every element of the 'Input Array' has been processed the 'Finished' signal
// is activated. All of this will happen in a single execution of the node, giving you
// the ability to evaluate a downstream graph multiple times with different inputs coming
// from the changing 'Element' output.
class OgnForEachDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnForEach.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnForEach.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnForEach.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnForEach.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnForEach;
    static std::tuple<int, int, int>sm_generatorVersionOgnForEach;
    static std::tuple<int, int, int>sm_targetVersionOgnForEach;
    static constexpr size_t staticAttributeCount = 8;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : arrayIn{offset}
        , execIn{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleInput<const ogn::RuntimeAttribute<ogn::kOgnInput, ogn::kCpu>,ogn::kCpu> arrayIn;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> execIn;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : arrayIndex{offset}
        , element{offset}
        , finished{offset,AttributeRole::eExecution}
        , loopBody{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<int,ogn::kCpu> arrayIndex;
        ogn::SimpleOutput<ogn::RuntimeAttribute<ogn::kOgnOutput, ogn::kCpu>,ogn::kCpu> element;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> finished;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> loopBody;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnForEachDatabase(NodeObj const& nodeObjParam)
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
    OgnForEachDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnForEachDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnForEachDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::execIn.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::arrayIndex.m_token, outputs::finished.m_token, outputs::loopBody.m_token
                )
            , kAccordingToContextIndex);
            inputs.execIn.setContext(contextObj);
            inputs.execIn.setHandle(std::get<0>(inputDataHandles0));
            outputs.arrayIndex.setContext(contextObj);
            outputs.arrayIndex.setHandle(std::get<0>(outputDataHandles0));
            outputs.finished.setContext(contextObj);
            outputs.finished.setHandle(std::get<1>(outputDataHandles0));
            outputs.loopBody.setContext(contextObj);
            outputs.loopBody.setHandle(std::get<2>(outputDataHandles0));
        }
        {
            ConstAttributeDataHandle __h;
            AttributeObj __a;
            __a = nodeObj.iNode->getAttributeByToken(nodeObj, inputs::arrayIn.m_token);
            __h = __a.iAttribute->getConstAttributeDataHandle(__a, kAccordingToContextIndex);
            const_cast<typename std::remove_const_t<ogn::RuntimeAttribute<ogn::kOgnInput, ogn::kCpu>&>>(inputs.arrayIn()).reset(contextObj, __h, __a);

        }
        {
            AttributeDataHandle __h;
            AttributeObj __a;
            __a = nodeObj.iNode->getAttributeByToken(nodeObj, outputs::element.m_token);
            __h = __a.iAttribute->getAttributeDataHandle(__a, kAccordingToContextIndex);
            outputs.element().reset(contextObj, __h, __a);

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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.ForEach");
            return;
        }
        auto& iToken{ *iTokenPtr };

        inputs::arrayIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::execIn.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::arrayIndex.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::element.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::finished.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::loopBody.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "For Each Loop");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,flowControl");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Activates the 'Loop Body' signal once for each element in the 'Input Array', making the current array member available in 'Element' with its index in 'Array Index'. After every element of the 'Input Array' has been processed the 'Finished' signal is activated. All of this will happen in a single execution of the node, giving you the ability to evaluate a downstream graph multiple times with different inputs coming from the changing 'Element' output.");
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
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataIconPath, "/isaac-sim/kit/cache/ogn_generated/1.79.2/omni.graph.action_nodes-1.50.4/omni.graph.action_nodes/ogn/icons/omni.graph.action.ForEach.svg");
    }
    static void initialize(const GraphContextObj&, const NodeObj& nodeObj)
    {
        const INode* iNode = nodeObj.iNode;
        AttributeObj attr;
        attr = iNode->getAttributeByToken(nodeObj, inputs::arrayIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The array to loop over");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Input Array");
        attr = iNode->getAttributeByToken(nodeObj, inputs::execIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Signal to the graph that this node is ready to be executed.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::arrayIndex.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The value of the current index being visited by the loop. Keeps the value of the last index\nafter the loop has completed. The index starts at zero and increments by one as it\nwalks through the members of 'Input Array'.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::element.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The current member of 'Input Array' being visited by the loop. Keeps the value of the last\narray element after the loop has completed.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::finished.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the final element of 'Input Array' has been processed signal the graph that\nexecution can continue downstream.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::loopBody.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "For each member of 'Input Array' signal the graph that execution can continue downstream.");
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
        sm_stateManagerOgnForEach.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.arrayIn().resolved()
            && inputs.execIn.isValid()
            && outputs.arrayIndex.isValid()
            && outputs.element().resolved()
            && outputs.finished.isValid()
            && outputs.loopBody.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.arrayIn.invalidateCachedPointer();
            inputs.execIn.invalidateCachedPointer();
            outputs.arrayIndex.invalidateCachedPointer();
            outputs.element.invalidateCachedPointer();
            outputs.finished.invalidateCachedPointer();
            outputs.loopBody.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::arrayIn.m_token) {
                inputs.arrayIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::execIn.m_token) {
                inputs.execIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::arrayIndex.m_token) {
                outputs.arrayIndex.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::element.m_token) {
                outputs.element.invalidateCachedPointer();
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
        if( !inputs.arrayIn.canVectorize()
            || !inputs.execIn.canVectorize()
            || !outputs.arrayIndex.canVectorize()
            || !outputs.element.canVectorize()
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
        if(inputs::arrayIn.m_token == token) {
            inputs.arrayIn.fetchDetails(attr);
            return;
        }
        if(outputs::element.m_token == token) {
            outputs.element.fetchDetails(attr);
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
    void onDynamicAttributesChanged(AttributeObj const& attribute, bool isAttributeCreated) {
        onDynamicAttributeCreatedOrRemoved(m_dynamicInputs, m_dynamicOutputs, m_dynamicStates, attribute, isAttributeCreated);
    }
    void onDataLocationChanged(AttributeObj const& attr) {
        if (! attr.isValid()) return;
        updateMappedAttributes(m_mappedAttributes, attr);
        NameToken token = attr.iAttribute->getNameToken(attr);
        if(token == inputs::arrayIn.m_token) {
            inputs.arrayIn.fetchDetails(attr);
            return;
        }
        if(token == inputs::execIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.execIn.setHandle(hdl);
            return;
        }
        if(token == outputs::arrayIndex.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.arrayIndex.setHandle(hdl);
            return;
        }
        if(token == outputs::element.m_token) {
            outputs.element.fetchDetails(attr);
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
ogn::StateManager OgnForEachDatabase::sm_stateManagerOgnForEach;
std::tuple<int, int, int> OgnForEachDatabase::sm_generatorVersionOgnForEach{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnForEachDatabase::sm_targetVersionOgnForEach{std::make_tuple(2,184,5)};
}
using namespace IOgnForEach;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnForEach, OgnForEachDatabase> s_registration("omni.graph.action.ForEach", 2, "omni.graph.action_nodes"); \
}
