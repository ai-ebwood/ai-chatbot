enum MessageRole {
  human,
  ai,
}
class BaseMessage {
  role: MessageRole;
  content: string;

  constructor(role: MessageRole, content: string) {
    this.role = role;
    this.content = content;
  }

  isHuman() {
    return this.role == MessageRole.human;
  }

  isAI() {
    return this.role == MessageRole.human;
  }
}

class HumanMessage extends BaseMessage {
  constructor(content: string) {
    super(MessageRole.human, content);
  }
}

class AIMessage extends BaseMessage {
  constructor(content: string) {
    super(MessageRole.ai, content);
  }
}

export { BaseMessage, MessageRole, HumanMessage, AIMessage };
