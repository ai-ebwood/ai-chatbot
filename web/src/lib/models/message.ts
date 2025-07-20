enum MessageRole {
  human,
  ai,
}
class BaseMessage {
  role: MessageRole;
  content: string;
  is_loading: boolean;

  constructor(role: MessageRole, content: string, is_loading: boolean = false) {
    this.role = role;
    this.content = content;
    this.is_loading = is_loading;
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
  constructor(content: string, is_loading: boolean = false) {
    super(MessageRole.ai, content, is_loading);
  }
}

export { BaseMessage, MessageRole, HumanMessage, AIMessage };
