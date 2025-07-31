# AI-Chatbot

## 使用langgraph实现AI-Chatbot的记忆系统

1. Checkpointer: 保存消息记录，只保存最近几轮的消息messages
2. Saver: 从每轮记录中提取用户信息UserPreference，使用langmem的create_memory_store_manager
3. rag vector_store: 每轮消息，将它的语义内容存储到向量数据库，metadata链接到消息数据库db
4. db: PostgresChatMessageHistoryWithId: 改进版的PostGresChatMessageHistory, 添加了message_id列，便于从vector_store索引具体消息

## 工作流程

1. 请求流程： 用户输入HumanMessage -> 从Saver中获取用户相关信息memories + 从vector_store提取相关历史聊天记录语义，然后从db提取消息histories + 最近几轮的消息，共同组成LLM的请求参数： memories + histories + messages
2. 得到结果: 从本轮消息中提取用户信息，保存到Saver；将本轮消息向量和元数据保存到vector_store; 保存本轮消息到db

## 有待改进

1. rag vector_store只用了默认的Qdrant检索，没有对结果作进一步过滤，可以使用re-rank提取出更精确的内容；同时要顾虑到时间的排序，时间越近的信息越有意义
2. Saver提取的用户信息，不够精确，有些无关信息也被保存了，可以使用更优质的prompt来提示
3. 消息历史语义信息，可以比如100轮之后，调用一次LLM来提取出里面的信息保存成summary，把这100轮信息archive，后续查询就只在这个summary查询而不用管原始的向量了。因为随着时间越久，历史信息只需要个大概就可以了，无需具体的情景重现。
4. 工作流程2中根据结果进行提取，应该放到redis/kafka等队列中，防止应用奔溃导致消息丢失；响应时间优化：LLM流式 + store/vector_store响应时间
