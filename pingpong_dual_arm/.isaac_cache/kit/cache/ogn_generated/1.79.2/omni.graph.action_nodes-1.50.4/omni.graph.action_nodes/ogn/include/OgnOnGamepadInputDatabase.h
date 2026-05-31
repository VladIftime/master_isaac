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

namespace OgnOnGamepadInputAttributes
{
namespace inputs
{
using gamepadElementIn_t = const NameToken&;
ogn::AttributeInitializer<const NameToken, ogn::kOgnInput> gamepadElementIn("inputs:gamepadElementIn", "token", kExtendedAttributeType_Regular);
using gamepadId_t = const uint32_t&;
ogn::AttributeInitializer<const uint32_t, ogn::kOgnInput> gamepadId("inputs:gamepadId", "uint", kExtendedAttributeType_Regular, 0);
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
}
namespace state
{
}
}
using namespace OgnOnGamepadInputAttributes;
namespace IOgnOnGamepadInput
{
// Event node which fires when a gamepad event occurs. This node only capture events
// on buttons, excluding triggers and sticks.
class OgnOnGamepadInputDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    struct TokenManager
    {
        NameToken FaceButtonBottom;
        NameToken FaceButtonRight;
        NameToken FaceButtonLeft;
        NameToken FaceButtonTop;
        NameToken LeftShoulder;
        NameToken RightShoulder;
        NameToken SpecialLeft;
        NameToken SpecialRight;
        NameToken LeftStickButton;
        NameToken RightStickButton;
        NameToken DpadUp;
        NameToken DpadRight;
        NameToken DpadDown;
        NameToken DpadLeft;
    };
    static TokenManager tokens;
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnGamepadInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnGamepadInput.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnGamepadInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnGamepadInput.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnGamepadInput;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnGamepadInput;
    static std::tuple<int, int, int>sm_targetVersionOgnOnGamepadInput;
    static constexpr size_t staticAttributeCount = 8;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : gamepadElementIn{offset}
        , gamepadId{offset}
        , onlyPlayback{offset}
        {}
        ogn::SimpleInput<const NameToken,ogn::kCpu> gamepadElementIn;
        ogn::SimpleInput<const uint32_t,ogn::kCpu> gamepadId;
        ogn::SimpleInput<const bool,ogn::kCpu> onlyPlayback;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : isPressed{offset}
        , pressed{offset,AttributeRole::eExecution}
        , released{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<bool,ogn::kCpu> isPressed;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> pressed;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> released;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnGamepadInputDatabase(NodeObj const& nodeObjParam)
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
    OgnOnGamepadInputDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnGamepadInputDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnGamepadInputDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::gamepadElementIn.m_token, inputs::gamepadId.m_token, inputs::onlyPlayback.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::isPressed.m_token, outputs::pressed.m_token, outputs::released.m_token
                )
            , kAccordingToContextIndex);
            inputs.gamepadElementIn.setContext(contextObj);
            inputs.gamepadElementIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.gamepadId.setContext(contextObj);
            inputs.gamepadId.setHandle(std::get<1>(inputDataHandles0));
            inputs.onlyPlayback.setContext(contextObj);
            inputs.onlyPlayback.setHandle(std::get<2>(inputDataHandles0));
            outputs.isPressed.setContext(contextObj);
            outputs.isPressed.setHandle(std::get<0>(outputDataHandles0));
            outputs.pressed.setContext(contextObj);
            outputs.pressed.setHandle(std::get<1>(outputDataHandles0));
            outputs.released.setContext(contextObj);
            outputs.released.setHandle(std::get<2>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnGamepadInput");
            return;
        }
        auto& iToken{ *iTokenPtr };
        OgnOnGamepadInputDatabase::tokens.FaceButtonBottom = iToken.getHandle("Face Button Bottom");
        OgnOnGamepadInputDatabase::tokens.FaceButtonRight = iToken.getHandle("Face Button Right");
        OgnOnGamepadInputDatabase::tokens.FaceButtonLeft = iToken.getHandle("Face Button Left");
        OgnOnGamepadInputDatabase::tokens.FaceButtonTop = iToken.getHandle("Face Button Top");
        OgnOnGamepadInputDatabase::tokens.LeftShoulder = iToken.getHandle("Left Shoulder");
        OgnOnGamepadInputDatabase::tokens.RightShoulder = iToken.getHandle("Right Shoulder");
        OgnOnGamepadInputDatabase::tokens.SpecialLeft = iToken.getHandle("Special Left");
        OgnOnGamepadInputDatabase::tokens.SpecialRight = iToken.getHandle("Special Right");
        OgnOnGamepadInputDatabase::tokens.LeftStickButton = iToken.getHandle("Left Stick Button");
        OgnOnGamepadInputDatabase::tokens.RightStickButton = iToken.getHandle("Right Stick Button");
        OgnOnGamepadInputDatabase::tokens.DpadUp = iToken.getHandle("D-Pad Up");
        OgnOnGamepadInputDatabase::tokens.DpadRight = iToken.getHandle("D-Pad Right");
        OgnOnGamepadInputDatabase::tokens.DpadDown = iToken.getHandle("D-Pad Down");
        OgnOnGamepadInputDatabase::tokens.DpadLeft = iToken.getHandle("D-Pad Left");

        static omni::fabric::Token inputs_gamepadElementIn_token {"Face Button Bottom"};
        inputs::gamepadElementIn.setDefault(inputs_gamepadElementIn_token.asTokenC());
        inputs::gamepadElementIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::gamepadId.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::onlyPlayback.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::isPressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::pressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::released.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On Gamepad Input");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,input:gamepad");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Event node which fires when a gamepad event occurs. This node only capture events on buttons, excluding triggers and sticks.");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExclusions, "tests");
        auto __schedulingInfo = nodeTypeObj.iNodeType->getSchedulingHints(nodeTypeObj);
        CARB_ASSERT(__schedulingInfo, "Could not acquire the scheduling hints");
        if (__schedulingInfo)
        {
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::gamepadElementIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The gamepad button that will trigger the downstream execution.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Gamepad Element In");
        attr.iAttribute->setMetadata(attr, "displayGroup", "parameters");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokens, "Face Button Bottom,Face Button Right,Face Button Left,Face Button Top,Left Shoulder,Right Shoulder,Special Left,Special Right,Left Stick Button,Right Stick Button,D-Pad Up,D-Pad Right,D-Pad Down,D-Pad Left");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokensRaw, "{\"FaceButtonBottom\": \"Face Button Bottom\", \"FaceButtonRight\": \"Face Button Right\", \"FaceButtonLeft\": \"Face Button Left\", \"FaceButtonTop\": \"Face Button Top\", \"LeftShoulder\": \"Left Shoulder\", \"RightShoulder\": \"Right Shoulder\", \"SpecialLeft\": \"Special Left\", \"SpecialRight\": \"Special Right\", \"LeftStickButton\": \"Left Stick Button\", \"RightStickButton\": \"Right Stick Button\", \"DpadUp\": \"D-Pad Up\", \"DpadRight\": \"D-Pad Right\", \"DpadDown\": \"D-Pad Down\", \"DpadLeft\": \"D-Pad Left\"}");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "\"Face Button Bottom\"");
        attr = iNode->getAttributeByToken(nodeObj, inputs::gamepadId.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "Gamepad id number starting from 0. Each gamepad will be registered automatically with a unique ID\nmonotonically increasing in the order they are connected. If a gamepad is disconnected, the ID\nassigned to the remaining gamepad will be adjusted accordingly so the IDs are always continuous\nand start from 0. Changing this value to a non-existing ID will result in an error prompt in the\nconsole and the node will not listen to any gamepad input.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Gamepad ID");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "0");
        attr = iNode->getAttributeByToken(nodeObj, inputs::onlyPlayback.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the node is only executed while the Stage is being played.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Only Simulate On Play");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "true");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isPressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "True if the gamepad button was pressed, False if it was released.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::pressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the gamepad element was pressed signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Pressed");
        attr = iNode->getAttributeByToken(nodeObj, outputs::released.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When the gamepad element was released signal to the graph that execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Released");
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
        sm_stateManagerOgnOnGamepadInput.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.gamepadElementIn.isValid()
            && inputs.gamepadId.isValid()
            && inputs.onlyPlayback.isValid()
            && outputs.isPressed.isValid()
            && outputs.pressed.isValid()
            && outputs.released.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.gamepadElementIn.invalidateCachedPointer();
            inputs.gamepadId.invalidateCachedPointer();
            inputs.onlyPlayback.invalidateCachedPointer();
            outputs.isPressed.invalidateCachedPointer();
            outputs.pressed.invalidateCachedPointer();
            outputs.released.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::gamepadElementIn.m_token) {
                inputs.gamepadElementIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::gamepadId.m_token) {
                inputs.gamepadId.invalidateCachedPointer();
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
        if( !inputs.gamepadElementIn.canVectorize()
            || !inputs.gamepadId.canVectorize()
            || !inputs.onlyPlayback.canVectorize()
            || !outputs.isPressed.canVectorize()
            || !outputs.pressed.canVectorize()
            || !outputs.released.canVectorize()
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
        if(token == inputs::gamepadElementIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.gamepadElementIn.setHandle(hdl);
            return;
        }
        if(token == inputs::gamepadId.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.gamepadId.setHandle(hdl);
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
ogn::StateManager OgnOnGamepadInputDatabase::sm_stateManagerOgnOnGamepadInput;
std::tuple<int, int, int> OgnOnGamepadInputDatabase::sm_generatorVersionOgnOnGamepadInput{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnGamepadInputDatabase::sm_targetVersionOgnOnGamepadInput{std::make_tuple(2,184,5)};
OgnOnGamepadInputDatabase::TokenManager OgnOnGamepadInputDatabase::tokens;
}
using namespace IOgnOnGamepadInput;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnGamepadInput, OgnOnGamepadInputDatabase> s_registration("omni.graph.action.OnGamepadInput", 2, "omni.graph.action_nodes"); \
}
