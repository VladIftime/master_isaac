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

namespace OgnOnKeyboardInputAttributes
{
namespace inputs
{
using altIn_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> altIn("inputs:altIn", "bool", kExtendedAttributeType_Regular, false);
using ctrlIn_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> ctrlIn("inputs:ctrlIn", "bool", kExtendedAttributeType_Regular, false);
using keyIn_t = const NameToken&;
ogn::AttributeInitializer<const NameToken, ogn::kOgnInput> keyIn("inputs:keyIn", "token", kExtendedAttributeType_Regular);
using onlyPlayback_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> onlyPlayback("inputs:onlyPlayback", "bool", kExtendedAttributeType_Regular, true);
using shiftIn_t = const bool&;
ogn::AttributeInitializer<const bool, ogn::kOgnInput> shiftIn("inputs:shiftIn", "bool", kExtendedAttributeType_Regular, false);
}
namespace outputs
{
using isPressed_t = bool&;
ogn::AttributeInitializer<bool, ogn::kOgnOutput> isPressed("outputs:isPressed", "bool", kExtendedAttributeType_Regular);
using keyOut_t = NameToken&;
ogn::AttributeInitializer<NameToken, ogn::kOgnOutput> keyOut("outputs:keyOut", "token", kExtendedAttributeType_Regular);
using pressed_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> pressed("outputs:pressed", "execution", kExtendedAttributeType_Regular);
using released_t = uint32_t&;
ogn::AttributeInitializer<uint32_t, ogn::kOgnOutput> released("outputs:released", "execution", kExtendedAttributeType_Regular);
}
namespace state
{
}
}
using namespace OgnOnKeyboardInputAttributes;
namespace IOgnOnKeyboardInput
{
// Event node which fires when a keyboard event occurs. The event can be any of the
// keys accepted by 'Key In', plus any combination of modifiers as specified by inputs
// 'Shift', 'Alt', and 'Ctrl'.
// For key combinations, the press event requires all modifiers to be held, with the
// 'Key In' pressed last. The release event is only triggered once when one of the chosen
// keys released after the pressed event happens.
// For example: if the combination is Ctrl-Shift-D, the pressed event happens once right
// after D is pressed while both Ctrl and Shift are held. The release event happens
// only once, when the user releases any one of Ctrl, Shift and D while holding them.
class OgnOnKeyboardInputDatabase : public omni::graph::core::ogn::OmniGraphDatabase
{
public:
    struct TokenManager
    {
        NameToken A;
        NameToken B;
        NameToken C;
        NameToken D;
        NameToken E;
        NameToken F;
        NameToken G;
        NameToken H;
        NameToken I;
        NameToken J;
        NameToken K;
        NameToken L;
        NameToken M;
        NameToken N;
        NameToken O;
        NameToken P;
        NameToken Q;
        NameToken R;
        NameToken S;
        NameToken T;
        NameToken U;
        NameToken V;
        NameToken W;
        NameToken X;
        NameToken Y;
        NameToken Z;
        NameToken Apostrophe;
        NameToken Backslash;
        NameToken Backspace;
        NameToken CapsLock;
        NameToken Comma;
        NameToken Del;
        NameToken Down;
        NameToken End;
        NameToken Enter;
        NameToken Equal;
        NameToken Escape;
        NameToken F1;
        NameToken F10;
        NameToken F11;
        NameToken F12;
        NameToken F2;
        NameToken F3;
        NameToken F4;
        NameToken F5;
        NameToken F6;
        NameToken F7;
        NameToken F8;
        NameToken F9;
        NameToken GraveAccent;
        NameToken Home;
        NameToken Insert;
        NameToken Key0;
        NameToken Key1;
        NameToken Key2;
        NameToken Key3;
        NameToken Key4;
        NameToken Key5;
        NameToken Key6;
        NameToken Key7;
        NameToken Key8;
        NameToken Key9;
        NameToken Left;
        NameToken LeftAlt;
        NameToken LeftBracket;
        NameToken LeftControl;
        NameToken LeftShift;
        NameToken LeftSuper;
        NameToken Menu;
        NameToken Minus;
        NameToken NumLock;
        NameToken Numpad0;
        NameToken Numpad1;
        NameToken Numpad2;
        NameToken Numpad3;
        NameToken Numpad4;
        NameToken Numpad5;
        NameToken Numpad6;
        NameToken Numpad7;
        NameToken Numpad8;
        NameToken Numpad9;
        NameToken NumpadAdd;
        NameToken NumpadDel;
        NameToken NumpadDivide;
        NameToken NumpadEnter;
        NameToken NumpadEqual;
        NameToken NumpadMultiply;
        NameToken NumpadSubtract;
        NameToken PageDown;
        NameToken PageUp;
        NameToken Pause;
        NameToken Period;
        NameToken PrintScreen;
        NameToken Right;
        NameToken RightAlt;
        NameToken RightBracket;
        NameToken RightControl;
        NameToken RightShift;
        NameToken RightSuper;
        NameToken ScrollLock;
        NameToken Semicolon;
        NameToken Slash;
        NameToken Space;
        NameToken Tab;
        NameToken Up;
    };
    static TokenManager tokens;
    template <typename StateInformation>
    CARB_DEPRECATED("sInternalState is deprecated. Use sSharedState or sPerInstanceState instead")
    static StateInformation& sInternalState(const NodeObj& nodeObj, InstanceIndex index = {kAuthoringGraphIndex}) {
        return sm_stateManagerOgnOnKeyboardInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sSharedState(const NodeObj& nodeObj) {
        return sm_stateManagerOgnOnKeyboardInput.getState<StateInformation>(nodeObj.nodeHandle, kAuthoringGraphIndex);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, InstanceIndex index) {
        return sm_stateManagerOgnOnKeyboardInput.getState<StateInformation>(nodeObj.nodeHandle, index);
    }
    template <typename StateInformation>
    static StateInformation& sPerInstanceState(const NodeObj& nodeObj, GraphInstanceID instanceId) {
        return sm_stateManagerOgnOnKeyboardInput.getState<StateInformation>(nodeObj.nodeHandle, instanceId);
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
    static ogn::StateManager sm_stateManagerOgnOnKeyboardInput;
    static std::tuple<int, int, int>sm_generatorVersionOgnOnKeyboardInput;
    static std::tuple<int, int, int>sm_targetVersionOgnOnKeyboardInput;
    static constexpr size_t staticAttributeCount = 11;
    std::vector<ogn::DynamicInput> m_dynamicInputs;
    std::vector<ogn::DynamicOutput> m_dynamicOutputs;
    std::vector<ogn::DynamicState> m_dynamicStates;
    std::vector<NameToken> m_mappedAttributes;
    bool m_canCachePointers{true};

    struct inputsT {
        inputsT(size_t const& offset)
        : altIn{offset}
        , ctrlIn{offset}
        , keyIn{offset}
        , onlyPlayback{offset}
        , shiftIn{offset}
        {}
        ogn::SimpleInput<const bool,ogn::kCpu> altIn;
        ogn::SimpleInput<const bool,ogn::kCpu> ctrlIn;
        ogn::SimpleInput<const NameToken,ogn::kCpu> keyIn;
        ogn::SimpleInput<const bool,ogn::kCpu> onlyPlayback;
        ogn::SimpleInput<const bool,ogn::kCpu> shiftIn;
    } inputs;

    struct outputsT {
        outputsT(size_t const& offset)
        : isPressed{offset}
        , keyOut{offset}
        , pressed{offset,AttributeRole::eExecution}
        , released{offset,AttributeRole::eExecution}
        {}
        ogn::SimpleOutput<bool,ogn::kCpu> isPressed;
        ogn::SimpleOutput<NameToken,ogn::kCpu> keyOut;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> pressed;
        ogn::SimpleOutput<uint32_t,ogn::kCpu> released;
    } outputs;

    //Only use this constructor for temporary stack-allocated object:
    OgnOnKeyboardInputDatabase(NodeObj const& nodeObjParam)
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
    OgnOnKeyboardInputDatabase(GraphContextObj const&, NodeObj const& nodeObjParam)
    : OgnOnKeyboardInputDatabase(nodeObjParam)
    {}

    //Main constructor
    OgnOnKeyboardInputDatabase(GraphContextObj const* contextObjParam, NodeObj const* nodeObjParam, size_t handleCount)
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
                ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle, ConstAttributeDataHandle,
                ConstAttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    inputs::altIn.m_token, inputs::ctrlIn.m_token, inputs::keyIn.m_token, inputs::onlyPlayback.m_token,
                    inputs::shiftIn.m_token
                )
            , kAccordingToContextIndex);
            auto outputDataHandles0 = getAttributesW<
                AttributeDataHandle, AttributeDataHandle, AttributeDataHandle, AttributeDataHandle
                >(contextObj, nodeObj.nodeContextHandle, std::make_tuple(
                    outputs::isPressed.m_token, outputs::keyOut.m_token, outputs::pressed.m_token, outputs::released.m_token
                )
            , kAccordingToContextIndex);
            inputs.altIn.setContext(contextObj);
            inputs.altIn.setHandle(std::get<0>(inputDataHandles0));
            inputs.ctrlIn.setContext(contextObj);
            inputs.ctrlIn.setHandle(std::get<1>(inputDataHandles0));
            inputs.keyIn.setContext(contextObj);
            inputs.keyIn.setHandle(std::get<2>(inputDataHandles0));
            inputs.onlyPlayback.setContext(contextObj);
            inputs.onlyPlayback.setHandle(std::get<3>(inputDataHandles0));
            inputs.shiftIn.setContext(contextObj);
            inputs.shiftIn.setHandle(std::get<4>(inputDataHandles0));
            outputs.isPressed.setContext(contextObj);
            outputs.isPressed.setHandle(std::get<0>(outputDataHandles0));
            outputs.keyOut.setContext(contextObj);
            outputs.keyOut.setHandle(std::get<1>(outputDataHandles0));
            outputs.pressed.setContext(contextObj);
            outputs.pressed.setHandle(std::get<2>(outputDataHandles0));
            outputs.released.setContext(contextObj);
            outputs.released.setHandle(std::get<3>(outputDataHandles0));
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
            CARB_LOG_ERROR("IToken not found when initializing omni.graph.action.OnKeyboardInput");
            return;
        }
        auto& iToken{ *iTokenPtr };
        OgnOnKeyboardInputDatabase::tokens.A = iToken.getHandle("A");
        OgnOnKeyboardInputDatabase::tokens.B = iToken.getHandle("B");
        OgnOnKeyboardInputDatabase::tokens.C = iToken.getHandle("C");
        OgnOnKeyboardInputDatabase::tokens.D = iToken.getHandle("D");
        OgnOnKeyboardInputDatabase::tokens.E = iToken.getHandle("E");
        OgnOnKeyboardInputDatabase::tokens.F = iToken.getHandle("F");
        OgnOnKeyboardInputDatabase::tokens.G = iToken.getHandle("G");
        OgnOnKeyboardInputDatabase::tokens.H = iToken.getHandle("H");
        OgnOnKeyboardInputDatabase::tokens.I = iToken.getHandle("I");
        OgnOnKeyboardInputDatabase::tokens.J = iToken.getHandle("J");
        OgnOnKeyboardInputDatabase::tokens.K = iToken.getHandle("K");
        OgnOnKeyboardInputDatabase::tokens.L = iToken.getHandle("L");
        OgnOnKeyboardInputDatabase::tokens.M = iToken.getHandle("M");
        OgnOnKeyboardInputDatabase::tokens.N = iToken.getHandle("N");
        OgnOnKeyboardInputDatabase::tokens.O = iToken.getHandle("O");
        OgnOnKeyboardInputDatabase::tokens.P = iToken.getHandle("P");
        OgnOnKeyboardInputDatabase::tokens.Q = iToken.getHandle("Q");
        OgnOnKeyboardInputDatabase::tokens.R = iToken.getHandle("R");
        OgnOnKeyboardInputDatabase::tokens.S = iToken.getHandle("S");
        OgnOnKeyboardInputDatabase::tokens.T = iToken.getHandle("T");
        OgnOnKeyboardInputDatabase::tokens.U = iToken.getHandle("U");
        OgnOnKeyboardInputDatabase::tokens.V = iToken.getHandle("V");
        OgnOnKeyboardInputDatabase::tokens.W = iToken.getHandle("W");
        OgnOnKeyboardInputDatabase::tokens.X = iToken.getHandle("X");
        OgnOnKeyboardInputDatabase::tokens.Y = iToken.getHandle("Y");
        OgnOnKeyboardInputDatabase::tokens.Z = iToken.getHandle("Z");
        OgnOnKeyboardInputDatabase::tokens.Apostrophe = iToken.getHandle("Apostrophe");
        OgnOnKeyboardInputDatabase::tokens.Backslash = iToken.getHandle("Backslash");
        OgnOnKeyboardInputDatabase::tokens.Backspace = iToken.getHandle("Backspace");
        OgnOnKeyboardInputDatabase::tokens.CapsLock = iToken.getHandle("CapsLock");
        OgnOnKeyboardInputDatabase::tokens.Comma = iToken.getHandle("Comma");
        OgnOnKeyboardInputDatabase::tokens.Del = iToken.getHandle("Del");
        OgnOnKeyboardInputDatabase::tokens.Down = iToken.getHandle("Down");
        OgnOnKeyboardInputDatabase::tokens.End = iToken.getHandle("End");
        OgnOnKeyboardInputDatabase::tokens.Enter = iToken.getHandle("Enter");
        OgnOnKeyboardInputDatabase::tokens.Equal = iToken.getHandle("Equal");
        OgnOnKeyboardInputDatabase::tokens.Escape = iToken.getHandle("Escape");
        OgnOnKeyboardInputDatabase::tokens.F1 = iToken.getHandle("F1");
        OgnOnKeyboardInputDatabase::tokens.F10 = iToken.getHandle("F10");
        OgnOnKeyboardInputDatabase::tokens.F11 = iToken.getHandle("F11");
        OgnOnKeyboardInputDatabase::tokens.F12 = iToken.getHandle("F12");
        OgnOnKeyboardInputDatabase::tokens.F2 = iToken.getHandle("F2");
        OgnOnKeyboardInputDatabase::tokens.F3 = iToken.getHandle("F3");
        OgnOnKeyboardInputDatabase::tokens.F4 = iToken.getHandle("F4");
        OgnOnKeyboardInputDatabase::tokens.F5 = iToken.getHandle("F5");
        OgnOnKeyboardInputDatabase::tokens.F6 = iToken.getHandle("F6");
        OgnOnKeyboardInputDatabase::tokens.F7 = iToken.getHandle("F7");
        OgnOnKeyboardInputDatabase::tokens.F8 = iToken.getHandle("F8");
        OgnOnKeyboardInputDatabase::tokens.F9 = iToken.getHandle("F9");
        OgnOnKeyboardInputDatabase::tokens.GraveAccent = iToken.getHandle("GraveAccent");
        OgnOnKeyboardInputDatabase::tokens.Home = iToken.getHandle("Home");
        OgnOnKeyboardInputDatabase::tokens.Insert = iToken.getHandle("Insert");
        OgnOnKeyboardInputDatabase::tokens.Key0 = iToken.getHandle("Key0");
        OgnOnKeyboardInputDatabase::tokens.Key1 = iToken.getHandle("Key1");
        OgnOnKeyboardInputDatabase::tokens.Key2 = iToken.getHandle("Key2");
        OgnOnKeyboardInputDatabase::tokens.Key3 = iToken.getHandle("Key3");
        OgnOnKeyboardInputDatabase::tokens.Key4 = iToken.getHandle("Key4");
        OgnOnKeyboardInputDatabase::tokens.Key5 = iToken.getHandle("Key5");
        OgnOnKeyboardInputDatabase::tokens.Key6 = iToken.getHandle("Key6");
        OgnOnKeyboardInputDatabase::tokens.Key7 = iToken.getHandle("Key7");
        OgnOnKeyboardInputDatabase::tokens.Key8 = iToken.getHandle("Key8");
        OgnOnKeyboardInputDatabase::tokens.Key9 = iToken.getHandle("Key9");
        OgnOnKeyboardInputDatabase::tokens.Left = iToken.getHandle("Left");
        OgnOnKeyboardInputDatabase::tokens.LeftAlt = iToken.getHandle("LeftAlt");
        OgnOnKeyboardInputDatabase::tokens.LeftBracket = iToken.getHandle("LeftBracket");
        OgnOnKeyboardInputDatabase::tokens.LeftControl = iToken.getHandle("LeftControl");
        OgnOnKeyboardInputDatabase::tokens.LeftShift = iToken.getHandle("LeftShift");
        OgnOnKeyboardInputDatabase::tokens.LeftSuper = iToken.getHandle("LeftSuper");
        OgnOnKeyboardInputDatabase::tokens.Menu = iToken.getHandle("Menu");
        OgnOnKeyboardInputDatabase::tokens.Minus = iToken.getHandle("Minus");
        OgnOnKeyboardInputDatabase::tokens.NumLock = iToken.getHandle("NumLock");
        OgnOnKeyboardInputDatabase::tokens.Numpad0 = iToken.getHandle("Numpad0");
        OgnOnKeyboardInputDatabase::tokens.Numpad1 = iToken.getHandle("Numpad1");
        OgnOnKeyboardInputDatabase::tokens.Numpad2 = iToken.getHandle("Numpad2");
        OgnOnKeyboardInputDatabase::tokens.Numpad3 = iToken.getHandle("Numpad3");
        OgnOnKeyboardInputDatabase::tokens.Numpad4 = iToken.getHandle("Numpad4");
        OgnOnKeyboardInputDatabase::tokens.Numpad5 = iToken.getHandle("Numpad5");
        OgnOnKeyboardInputDatabase::tokens.Numpad6 = iToken.getHandle("Numpad6");
        OgnOnKeyboardInputDatabase::tokens.Numpad7 = iToken.getHandle("Numpad7");
        OgnOnKeyboardInputDatabase::tokens.Numpad8 = iToken.getHandle("Numpad8");
        OgnOnKeyboardInputDatabase::tokens.Numpad9 = iToken.getHandle("Numpad9");
        OgnOnKeyboardInputDatabase::tokens.NumpadAdd = iToken.getHandle("NumpadAdd");
        OgnOnKeyboardInputDatabase::tokens.NumpadDel = iToken.getHandle("NumpadDel");
        OgnOnKeyboardInputDatabase::tokens.NumpadDivide = iToken.getHandle("NumpadDivide");
        OgnOnKeyboardInputDatabase::tokens.NumpadEnter = iToken.getHandle("NumpadEnter");
        OgnOnKeyboardInputDatabase::tokens.NumpadEqual = iToken.getHandle("NumpadEqual");
        OgnOnKeyboardInputDatabase::tokens.NumpadMultiply = iToken.getHandle("NumpadMultiply");
        OgnOnKeyboardInputDatabase::tokens.NumpadSubtract = iToken.getHandle("NumpadSubtract");
        OgnOnKeyboardInputDatabase::tokens.PageDown = iToken.getHandle("PageDown");
        OgnOnKeyboardInputDatabase::tokens.PageUp = iToken.getHandle("PageUp");
        OgnOnKeyboardInputDatabase::tokens.Pause = iToken.getHandle("Pause");
        OgnOnKeyboardInputDatabase::tokens.Period = iToken.getHandle("Period");
        OgnOnKeyboardInputDatabase::tokens.PrintScreen = iToken.getHandle("PrintScreen");
        OgnOnKeyboardInputDatabase::tokens.Right = iToken.getHandle("Right");
        OgnOnKeyboardInputDatabase::tokens.RightAlt = iToken.getHandle("RightAlt");
        OgnOnKeyboardInputDatabase::tokens.RightBracket = iToken.getHandle("RightBracket");
        OgnOnKeyboardInputDatabase::tokens.RightControl = iToken.getHandle("RightControl");
        OgnOnKeyboardInputDatabase::tokens.RightShift = iToken.getHandle("RightShift");
        OgnOnKeyboardInputDatabase::tokens.RightSuper = iToken.getHandle("RightSuper");
        OgnOnKeyboardInputDatabase::tokens.ScrollLock = iToken.getHandle("ScrollLock");
        OgnOnKeyboardInputDatabase::tokens.Semicolon = iToken.getHandle("Semicolon");
        OgnOnKeyboardInputDatabase::tokens.Slash = iToken.getHandle("Slash");
        OgnOnKeyboardInputDatabase::tokens.Space = iToken.getHandle("Space");
        OgnOnKeyboardInputDatabase::tokens.Tab = iToken.getHandle("Tab");
        OgnOnKeyboardInputDatabase::tokens.Up = iToken.getHandle("Up");

        static omni::fabric::Token inputs_keyIn_token {"A"};
        inputs::keyIn.setDefault(inputs_keyIn_token.asTokenC());
        inputs::altIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::ctrlIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::keyIn.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::onlyPlayback.initialize(iToken, *iNodeType, nodeTypeObj);
        inputs::shiftIn.initialize(iToken, *iNodeType, nodeTypeObj);

        outputs::isPressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::keyOut.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::pressed.initialize(iToken, *iNodeType, nodeTypeObj);
        outputs::released.initialize(iToken, *iNodeType, nodeTypeObj);

        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataExtension, "omni.graph.action_nodes");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataUiName, "On Keyboard Input");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataCategories, "graph:action,input:keyboard");
        iNodeType->setMetadata(nodeTypeObj, kOgnMetadataDescription, "Event node which fires when a keyboard event occurs. The event can be any of the keys accepted by 'Key In', plus any combination of modifiers as specified by inputs 'Shift', 'Alt', and 'Ctrl'.\nFor key combinations, the press event requires all modifiers to be held, with the 'Key In' pressed last. The release event is only triggered once when one of the chosen keys released after the pressed event happens.\nFor example: if the combination is Ctrl-Shift-D, the pressed event happens once right after D is pressed while both Ctrl and Shift are held. The release event happens only once, when the user releases any one of Ctrl, Shift and D while holding them.");
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
        attr = iNode->getAttributeByToken(nodeObj, inputs::altIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the Alt key modifier must be pressed along with the 'Key In' to activate the output.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Alt");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr = iNode->getAttributeByToken(nodeObj, inputs::ctrlIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the Ctrl key modifier must be pressed along with the 'Key In' to activate the output.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Ctrl");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr = iNode->getAttributeByToken(nodeObj, inputs::keyIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The key that triggers the downstream execution, not including modifiers.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Key In");
        attr.iAttribute->setMetadata(attr, "displayGroup", "parameters");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokens, "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,Apostrophe,Backslash,Backspace,CapsLock,Comma,Del,Down,End,Enter,Equal,Escape,F1,F10,F11,F12,F2,F3,F4,F5,F6,F7,F8,F9,GraveAccent,Home,Insert,Key0,Key1,Key2,Key3,Key4,Key5,Key6,Key7,Key8,Key9,Left,LeftAlt,LeftBracket,LeftControl,LeftShift,LeftSuper,Menu,Minus,NumLock,Numpad0,Numpad1,Numpad2,Numpad3,Numpad4,Numpad5,Numpad6,Numpad7,Numpad8,Numpad9,NumpadAdd,NumpadDel,NumpadDivide,NumpadEnter,NumpadEqual,NumpadMultiply,NumpadSubtract,PageDown,PageUp,Pause,Period,PrintScreen,Right,RightAlt,RightBracket,RightControl,RightShift,RightSuper,ScrollLock,Semicolon,Slash,Space,Tab,Up");
        attr.iAttribute->setMetadata(attr, kOgnMetadataAllowedTokensRaw, "[\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\", \"I\", \"J\", \"K\", \"L\", \"M\", \"N\", \"O\", \"P\", \"Q\", \"R\", \"S\", \"T\", \"U\", \"V\", \"W\", \"X\", \"Y\", \"Z\", \"Apostrophe\", \"Backslash\", \"Backspace\", \"CapsLock\", \"Comma\", \"Del\", \"Down\", \"End\", \"Enter\", \"Equal\", \"Escape\", \"F1\", \"F10\", \"F11\", \"F12\", \"F2\", \"F3\", \"F4\", \"F5\", \"F6\", \"F7\", \"F8\", \"F9\", \"GraveAccent\", \"Home\", \"Insert\", \"Key0\", \"Key1\", \"Key2\", \"Key3\", \"Key4\", \"Key5\", \"Key6\", \"Key7\", \"Key8\", \"Key9\", \"Left\", \"LeftAlt\", \"LeftBracket\", \"LeftControl\", \"LeftShift\", \"LeftSuper\", \"Menu\", \"Minus\", \"NumLock\", \"Numpad0\", \"Numpad1\", \"Numpad2\", \"Numpad3\", \"Numpad4\", \"Numpad5\", \"Numpad6\", \"Numpad7\", \"Numpad8\", \"Numpad9\", \"NumpadAdd\", \"NumpadDel\", \"NumpadDivide\", \"NumpadEnter\", \"NumpadEqual\", \"NumpadMultiply\", \"NumpadSubtract\", \"PageDown\", \"PageUp\", \"Pause\", \"Period\", \"PrintScreen\", \"Right\", \"RightAlt\", \"RightBracket\", \"RightControl\", \"RightShift\", \"RightSuper\", \"ScrollLock\", \"Semicolon\", \"Slash\", \"Space\", \"Tab\", \"Up\"]");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "\"A\"");
        attr = iNode->getAttributeByToken(nodeObj, inputs::onlyPlayback.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the node is only executed while the Stage is being played.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Only Simulate On Play");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr.iAttribute->setMetadata(attr, kOgnMetadataDefault, "true");
        attr = iNode->getAttributeByToken(nodeObj, inputs::shiftIn.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "When true, the Shift key modifier must be pressed along with the 'Key In' to activate the output.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Shift");
        attr.iAttribute->setMetadata(attr, kOgnMetadataLiteralOnly, "1");
        attr = iNode->getAttributeByToken(nodeObj, outputs::isPressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "True if the most recent activation was the key being pressed, False if it was released.");
        attr = iNode->getAttributeByToken(nodeObj, outputs::keyOut.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "The key that was pressed or released to trigger the execution of this node.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Key Out");
        attr = iNode->getAttributeByToken(nodeObj, outputs::pressed.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After the key was pressed, along with any required modifiers,\nsignal to the graph that the execution can continue downstream.");
        attr.iAttribute->setMetadata(attr, kOgnMetadataUiName, "Pressed");
        attr = iNode->getAttributeByToken(nodeObj, outputs::released.token());
        attr.iAttribute->setMetadata(attr, kOgnMetadataDescription, "After the key or any of the required modifiers were released,\nsignal to the graph that the execution can continue downstream.");
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
        sm_stateManagerOgnOnKeyboardInput.removeState(nodeObj.nodeHandle, instanceID);
    }
    bool validate() const {
        return validateNode()
            && inputs.altIn.isValid()
            && inputs.ctrlIn.isValid()
            && inputs.keyIn.isValid()
            && inputs.onlyPlayback.isValid()
            && inputs.shiftIn.isValid()
            && outputs.isPressed.isValid()
            && outputs.keyOut.isValid()
            && outputs.pressed.isValid()
            && outputs.released.isValid()
        ;
    }
    void preCompute() {
        if(m_canCachePointers == false) {
            inputs.altIn.invalidateCachedPointer();
            inputs.ctrlIn.invalidateCachedPointer();
            inputs.keyIn.invalidateCachedPointer();
            inputs.onlyPlayback.invalidateCachedPointer();
            inputs.shiftIn.invalidateCachedPointer();
            outputs.isPressed.invalidateCachedPointer();
            outputs.keyOut.invalidateCachedPointer();
            outputs.pressed.invalidateCachedPointer();
            outputs.released.invalidateCachedPointer();
            return;
        }
        for(NameToken const& attrib : m_mappedAttributes) {
            if(attrib == inputs::altIn.m_token) {
                inputs.altIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::ctrlIn.m_token) {
                inputs.ctrlIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::keyIn.m_token) {
                inputs.keyIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::onlyPlayback.m_token) {
                inputs.onlyPlayback.invalidateCachedPointer();
                continue;
            }
            if(attrib == inputs::shiftIn.m_token) {
                inputs.shiftIn.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::isPressed.m_token) {
                outputs.isPressed.invalidateCachedPointer();
                continue;
            }
            if(attrib == outputs::keyOut.m_token) {
                outputs.keyOut.invalidateCachedPointer();
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
        if( !inputs.altIn.canVectorize()
            || !inputs.ctrlIn.canVectorize()
            || !inputs.keyIn.canVectorize()
            || !inputs.onlyPlayback.canVectorize()
            || !inputs.shiftIn.canVectorize()
            || !outputs.isPressed.canVectorize()
            || !outputs.keyOut.canVectorize()
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
        if(token == inputs::altIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.altIn.setHandle(hdl);
            return;
        }
        if(token == inputs::ctrlIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.ctrlIn.setHandle(hdl);
            return;
        }
        if(token == inputs::keyIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.keyIn.setHandle(hdl);
            return;
        }
        if(token == inputs::onlyPlayback.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.onlyPlayback.setHandle(hdl);
            return;
        }
        if(token == inputs::shiftIn.m_token) {
            ConstAttributeDataHandle hdl = attr.iAttribute->getConstAttributeDataHandle(attr, m_offset);
            inputs.shiftIn.setHandle(hdl);
            return;
        }
        if(token == outputs::isPressed.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.isPressed.setHandle(hdl);
            return;
        }
        if(token == outputs::keyOut.m_token) {
            AttributeDataHandle hdl = attr.iAttribute->getAttributeDataHandle(attr, m_offset);
            outputs.keyOut.setHandle(hdl);
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
ogn::StateManager OgnOnKeyboardInputDatabase::sm_stateManagerOgnOnKeyboardInput;
std::tuple<int, int, int> OgnOnKeyboardInputDatabase::sm_generatorVersionOgnOnKeyboardInput{std::make_tuple(1,79,2)};
std::tuple<int, int, int> OgnOnKeyboardInputDatabase::sm_targetVersionOgnOnKeyboardInput{std::make_tuple(2,184,5)};
OgnOnKeyboardInputDatabase::TokenManager OgnOnKeyboardInputDatabase::tokens;
}
using namespace IOgnOnKeyboardInput;
#define REGISTER_OGN_NODE() \
namespace { \
    ogn::NodeTypeBootstrapImpl<OgnOnKeyboardInput, OgnOnKeyboardInputDatabase> s_registration("omni.graph.action.OnKeyboardInput", 4, "omni.graph.action_nodes"); \
}
