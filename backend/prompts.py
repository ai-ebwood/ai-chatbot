generate_result_prompt = """
根据提供的上下文summary, 请解答用户的问题question:

<summary>
{summary}
</summary>

用户的问题question:
"""

generate_summary_prompt = """
请根据已有的总结old_summary（可能为空), 加上最新的消息列表messages，得到新的总结.

已有的总结内容:
<old_summary>
{summary}
</old_summary>

消息列表:
<messages>
{messages}
</messages>

输出结果:
summary: 总结example
"""

# 与已有记忆相关的prompt
memory_prompt = """根据用户当前的相关记忆memories(可能为空)，回答用户的问题。

<memories>
{memories}
</memories>
"""

generate_memory_prompt = """
Update the existing user profile (or create a new one if it doesn't exist) based on the shared information."""
