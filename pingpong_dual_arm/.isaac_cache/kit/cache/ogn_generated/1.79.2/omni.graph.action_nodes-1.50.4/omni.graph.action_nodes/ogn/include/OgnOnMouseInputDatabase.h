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
#include <omni/graph/core/Type.h>
#include <omni/graph/core/ogn/SimpleAttribute.h>
#include <omni/graph/core/tuple.h>

namespace OgnOnMouseInputAttributes
{
namespace inputs
{
using mouseElement_t = const NameToken&;
ogn::AttributeInitializer<const NameToken, ogn::kOgnInput> mouseElement("inputs:mouseElement", "token", kExtendedAttributeType_Regular);
using onlyPlayback_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> onlyPlayback("inputs:onlyPlayback", "bool", kExtendedAttributeType_Regular, true);
}
namespace outputs
{
using isPressed_t = bool&;
ogn::AttributeInitializer<bool, ogn::kOgnOutput> isPressed("outputs:isPressed", "bool", kExtendedAttributeType_Regular);
using pressed_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> pressed("outputs:pressed", "execution", kExtendedAttributeType_Regular);
using released_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> released("outputs:released", "execution", kExtendedAttributeType_Regular);
using value_t = pxr::GfVec2f&;
ogn::AttributeInitializer<pxr::GfVec2f, ogn::kOgnOutput> value("outputs:value", "float2", kExtendedAttributeType_Regular);
using valueChanged_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> valueChanged("outputs:valueChanged", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnOnMouseInputAttributes;
namespace IOgnOnMouseInput
{
// Event node which fires when a mouse event occurs.
// You can choose which 'Mouse Element' this node reacts to. When 'Mouse Element' is
// chosen to be a button, the only meaningful outputs are: 'Pressed', 'Released' and
// 'Is Pressed'. When scroll or move events are chosen, the only meaningful outputs
// are: 'Value Changed' and 'Value'. You can choose to output normalized or pixel coordinates
// of the mouse.
// Pixel coordinates are the absolute position of the mouse cursor in pixel units. The
// original point is the upper left corner. The minimum value is 0, and the maximum
// value depends on the size of the window.
// Normalized coordinates are the relative position of the mouse cursor to the window.
// The value is always between 0 and 1.
class OgnOnMouseInputDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    struct TokenManager
    {
        NameToken LeftButton;
        NameToken MiddleButton;
        NameToken RightButton;
        NameToken NormalizedMove;
        NameToken PixelMove;
        NameToken Scroll;
    };
    static TokenManager tokens;
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnMouseInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnMouseInput.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnMouseInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnMouseInput.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnMouseInput;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnMouseInput;
    static std::tuple<int, int, int>sm_targetVersionOgnOnMouseInput;
    static constexpr size_t staticAttributeCount = 9;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : mouseElement{offset}
        , onlyPlayback{offset}
        {}
        ogn::SimpleInput<const NameToken,ogn::kCpu> mouseElement;
        ogn::SimpleInput<const bool,ogn::kCpu> onlyPlayback;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : isPressed{offset}
        , pressed{offset,AttributeRole::eExecution}
        , released{offset,AttributeRole::eExecution}
        , value{offset}
        , valueChanged{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<bool,ogn::kCpu> isPressed;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> pressed;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> released;
        ogn::SimpleOutput<pxr::GfVec2f,ogn::kCpu> value;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> valueChanged;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnMouseInputDatabase(NodeObj const& nodeObjParam)
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
    OgnOnMouseInputDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnMouseInputDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnMouseInputDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                    inputs::mouseElement.m_token, inputs::onlyPlayback.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle,
                AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::isPressed.m_token, outputs::pressed.m_token, outputs::released.m_token, outputs::value.m_token,
                    outputs::valueChanged.m_token
                )
            , kAccordingToContextIndex);
            inputs.mouseElement.setContext(contextObj);
            inputs.mouseElement.setHandle(std::get<0>(inputDataHandles0));
            inputs.onlyPlayback.setContext(contextObj);
            inputs.onlyPlayback.setHandle(std::get<1>(inputDataHandles0));
            outputs.isPressed.setContext(contextObj);
            outputs.isPressed.setHandle(std::get<0>(outputDataHandles0));
            outputs.pressed.setContext(contextObj);
            outputs.pressed.setHandle(std::get<1>(outputDataHandles0));
            outputs.released.setContext(contextObj);
            outputs.released.setHandle(std::get<2>(outputDataHandles0));
            outputs.value.setContext(contextObj);
            outputs.value.setHandle(std::get<3>(outputDataHandles0));
            outputs.valueChanged.setContext(contextObj);
            outputs.valueChanged.setHandle(std::get<4>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnMouseInput");
            return;
        }
        auto& iToken{ *iTokenPtr };
        OgnOnMouseInputDatabase::tokens.LeftButton = iToken.getHandle("Left Button");
        OgnOnMouseInputDatabase::tokens.MiddleButton = iToken.getHandle("Middle Button");
        OgnOnMouseInputDatabase::tokens.RightButton = iToken.getHandle("Right Button");
        OgnOnMouseInputDatabase::tokens.NormalizedMove = iToken.getHandle("Normalized Move");
        OgnOnMouseInputDatabase::tokens.PixelMove = iToken.getHandle("Pixel Move");
        OgnOnMouseInputDatabase::tokens.Scroll = iToken.getHandle("Scroll");

        static omni::fabric::Token inputs_mouseElement_token {"Left Button"};
        inputs::mouseElement.setDefault(inputs_mouseElement_token.asTokenC());
        inputs::mouseElement.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::onlyPlayback.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::isPressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::pressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::released.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::value.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::valueChanged.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On Mouse Input");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,input:mouse");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Event node which fires when a mouse event occurs.\nYou can choose which 'Mouse Element' this node reacts to. When 'Mouse Element' is chosen to be a button, the only meaningful outputs are: 'Pressed', 'Released' and 'Is Pressed'. When scroll or move events are chosen, the only meaningful outputs are: 'Value Changed' and 'Value'. You can choose to output normalized or pixel coordinates of the mouse.\nPixel coordinates are the absolute position of the mouse cursor in pixel units. The original point is the upper left corner. The minimum value is 0, and the maximum value depends on the size of the window.\nNormalized coordinates are the relative position of the mouse cursor to the window. The value is always between 0 and 1.");
        auto __schedulingInfo = nodeTypeObj.iNodeType->getSchedulingHints(nodeTypeObj);
        CARB_ASSERT(__schedulingInfo, "Could not acquire the scheduling hints");
        if (__schedulingInfo)
        {
            __schedulingInfo->setThreadSafety(eThreadSafety::eSafe);
            __schedulingInfo->setComputeRule(eComputeRule::eOnRequest);
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::mouseElement.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The name of the mouse event that will trigger the downstream execution.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Mouse Element");
        attr.iAttribute->setMetadata(attr, "displayGroup", "parameters");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokens, "Left Button,Middle Button,Right Button,Normalized Move,Pixel Move,Scroll");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokensRaw, "{\"LeftButton\": \"Left Button\", \"MiddleButton\": \"Middle Button\", \"RightButton\": \"Right Button\", \"NormalizedMove\": \"Normalized Move\", \"PixelMove\": \"Pixel Move\", \"Scroll\": \"Scroll\"}");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "\"Left Button\"");
        attr = iNode->getAttributeByToken(nodeObj, inputs::onlyPlayback.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the node is only executed while the Stage is being played.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Only Simulate On Play");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "true");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isPressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "True if the mouse button was pressed, False if it was released or 'Mouse Element' is not\nrelated to a button, as in a move or scroll.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::pressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When any mouse button was pressed signal to the graph that the execution can continue downstream.\nWill not execute on move or scroll events.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Pressed");
        attr = iNode->getAttributeByToken(nodeObj, outputs::released.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When any mouse button was released signal to the graph that the execution can continue downstream.\nWill not execute on move or scroll events.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Released");
        attr = iNode->getAttributeByToken(nodeObj, outputs::value.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The meaning of this output depends on 'Mouse Element'.\n\nNormalized Move: will output the normalized coordinates of mouse, each element of the vector is in the range of [0, 1].\n\nPixel Move: will output the absolute coordinates of mouse, each vector is in the range of [0, pixel width/height of the window].\n\nScroll: will output the change of scroll value.\n\nOtherwise: will output [0,0].");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Delta Value");
        attr = iNode->getAttributeByToken(nodeObj, outputs::valueChanged.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the mouse was moved or scrolled signal to the graph that the execution can continue downstream.\nWill not execute on press or release events.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Moved");
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
        sm_stateManagerOgnOnMouseInput.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.mouseElement.isValid()
            && inputs.onlyPlayback.isValid()
            && outputs.isPressed.isValid()
            && outputs.pressed.isValid()
            && outputs.released.isValid()
            && outputs.value.isValid()
            && outputs.valueChanged.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.mouseElement.invalidateCachedPointer();
            inputs.onlyPlayback.invalidateCachedPointer();
            outputs.isPressed.invalidateCachedPointer();
            outputs.pressed.invalidateCachedPointer();
            outputs.released.invalidateCachedPointer();
            outputs.value.invalidateCachedPointer();
            outputs.valueChanged.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::mouseElement.m_token) {
                inputs.mouseElement.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::onlyPlayback.m_token) {
                inputs.onlyPlayback.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::isPressed.m_token) {
                outputs.isPressed.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::pressed.m_token) {
                outputs.pressed.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::released.m_token) {
                outputs.released.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::value.m_token) {
                outputs.value.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::valueChanged.m_token) {
                outputs.valueChanged.invalidateCachedPointer();
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
        if( !inputs.mouseElement.canVectorize()
            || !inputs.onlyPlayback.canVectorize()
            || !outputs.isPressed.canVectorize()
            || !outputs.pressed.canVectorize()
            || !outputs.released.canVectorize()
            || !outputs.value.canVectorize()
            || !outputs.valueChanged.canVectorize()
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
        if(token == inputs::mouseElement.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.mouseElement.setHandle(hdl);
            return;
        }
        if(token == inputs::onlyPlayback.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.onlyPlayback.setHandle(hdl);
            return;
        }
        if(token == outputs::isPressed.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.isPressed.setHandle(hdl);
            return;
        }
        if(token == outputs::pressed.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.pressed.setHandle(hdl);
            return;
        }
        if(token == outputs::released.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.released.setHandle(hdl);
            return;
        }
        if(token == outputs::value.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.value.setHandle(hdl);
            return;
        }
        if(token == outputs::valueChanged.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.valueChanged.setHandle(hdl);
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
ogn::StateManager OgnOnMouseInputDatabase::sm_stateManagerOgnOnMouseInput;
std::tuple<int, int, int> OgnOnMouseInputDatabase::sm_generatorVersionOgnOnMouseInput{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnMouseInputDatabase::sm_targetVersionOgnOnMouseInput{std::make_tuple(2,184,5)};
OgnOnMouseInputDatabase::TokenManager OgnOnMouseInputDatabase::tokens;
}
using namespace IOgnOnMouseInput;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnMouseInput, OgnOnMouseInputDatabase> s_registration("omni.graph.action.OnMouseInput", 2, "omni.graph.action_nodes"); \
}
