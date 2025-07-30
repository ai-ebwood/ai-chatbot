enum MessageType {
  human,
  ai,
  error,
}

function stringToMessageType(role: string): MessageType {
  switch (role) {
    case "human":
      return MessageType.human;
    case "ai":
      return MessageType.ai;
    case "error":
      return MessageType.error;
    default:
      throw new Error(`Invalid message type: ${role}`);
  }
}
class BaseMessage {
  type: MessageType;
  content: string;
  is_loading: boolean;

  constructor(type: MessageType, content: string, is_loading: boolean = false) {
    this.type = type;
    this.content = content;
    this.is_loading = is_loading;
  }

  isHuman() {
    return this.type == MessageType.human;
  }

  isAI() {
    return this.type == MessageType.ai;
  }

  isError() {
    return this.type == MessageType.error;
  }
}

class HumanMessage extends BaseMessage {
  constructor(content: string) {
    super(MessageType.human, content);
  }
}

class AIMessage extends BaseMessage {
  constructor(content: string, is_loading: boolean = false) {
    super(MessageType.ai, content, is_loading);
  }
}

class ErrorMessage extends BaseMessage {
  constructor(content: string) {
    super(MessageType.error, content);
  }
}

export {
  BaseMessage,
  MessageType as MessageRole,
  HumanMessage,
  AIMessage,
  ErrorMessage,
  stringToMessageType
};
