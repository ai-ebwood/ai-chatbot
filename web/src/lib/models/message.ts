enum MessageRole {
  human,
  ai,
  error,
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
    return this.role == MessageRole.ai;
  }

  isError() {
    return this.role == MessageRole.error;
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

class ErrorMessage extends BaseMessage {
  constructor(content: string) {
    super(MessageRole.error, content);
  }
}

export { BaseMessage, MessageRole, HumanMessage, AIMessage, ErrorMessage };
