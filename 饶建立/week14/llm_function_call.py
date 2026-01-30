import json
import openai  # 或其他大模型API
from typing import Dict, Any


# 模拟一个简单的函数注册器
class FunctionManager:
    def __init__(self):
        self.functions = {}

    def register(self, func):
        """注册函数"""
        self.functions[func.__name__] = func
        return func

    def get_tools_schema(self):
        """生成工具schema（OpenAI格式）"""
        tools = []

        for name, func in self.functions.items():
            # 这里简化了，实际应该解析函数参数和文档
            if name == "get_weather":
                schema = {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取城市的天气信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "城市名称，如：北京、上海"
                                }
                            },
                            "required": ["city"]
                        }
                    }
                }
                tools.append(schema)

            elif name == "calculator":
                schema = {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "计算数学表达式",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "数学表达式，如：2+3*4"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }
                tools.append(schema)

        return tools

    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """执行函数"""
        return self.functions[function_name](**arguments)


# 创建函数管理器
fm = FunctionManager()


# 注册工具函数
@fm.register
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 这里模拟调用天气API
    weather_data = {
        "北京": "晴天，25°C，空气质量良",
        "上海": "多云，28°C，空气质量优",
        "广州": "阵雨，30°C，空气质量良"
    }
    return weather_data.get(city, "未知城市")


@fm.register
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 注意：生产环境要用更安全的方法
        return f"{expression} = {result}"
    except:
        return "计算失败"


# 模拟大模型API调用
class MockLLM:
    """模拟大模型API"""

    def chat_completion(self, messages, tools=None):
        """模拟大模型的响应"""

        user_input = messages[-1]["content"]

        # 模拟大模型思考：是否需要调用工具？
        if "天气" in user_input or "temperature" in user_input.lower():
            # 大模型决定调用 get_weather 函数
            city = "北京"  # 简单提取，实际大模型会智能提取
            if "上海" in user_input:
                city = "上海"
            elif "广州" in user_input:
                city = "广州"

            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": city})
                            }
                        }]
                    }
                }]
            }

        elif "计算" in user_input or any(op in user_input for op in ["+", "-", "*", "/"]):
            # 大模型决定调用 calculator 函数
            import re
            match = re.search(r'(\d+[\+\-\*/]\d+)', user_input)
            expression = match.group(1) if match else "2+2"

            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_456",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": expression})
                            }
                        }]
                    }
                }]
            }

        # 不需要工具调用，直接回复
        return {
            "choices": [{
                "message": {
                    "content": f"我收到了你的消息：'{user_input}'",
                    "tool_calls": None
                }
            }]
        }


# 主交互流程
def chat_with_tools(user_input: str):
    """完整的聊天流程"""

    llm = MockLLM()
    conversation_history = []

    print(f"\n👤 用户: {user_input}")

    # 1. 准备消息
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以调用工具来帮助用户。"},
        {"role": "user", "content": user_input}
    ]

    # 2. 第一次调用大模型（让模型决定是否调用工具）
    print("🤖 大模型思考中...")
    response = llm.chat_completion(
        messages=messages,
        tools=fm.get_tools_schema()  # 告诉模型有哪些工具可用
    )

    message = response["choices"][0]["message"]

    # 3. 检查是否调用了工具
    if message.get("tool_calls"):
        print("🛠️  大模型决定调用工具...")

        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            print(f"  调用工具: {func_name}")
            print(f"  参数: {arguments}")

            # 4. 执行工具函数
            try:
                tool_result = fm.execute(func_name, arguments)
                print(f"  工具执行结果: {tool_result}")

                # 5. 把结果发送回大模型
                messages.append(message)  # 添加模型的工具调用请求
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result)
                })

                # 6. 第二次调用大模型（让它基于工具结果生成回复）
                print("🤖 大模型基于工具结果生成回答...")
                final_response = llm.chat_completion(messages=messages)
                final_message = final_response["choices"][0]["message"]

                print(f"💬 助手: {final_message['content']}")
                return final_message['content']

            except Exception as e:
                print(f"❌ 工具执行失败: {e}")
                return f"工具执行失败: {e}"

    else:
        # 没有工具调用，直接回复
        print(f"💬 助手: {message['content']}")
        return message['content']


# 运行示例
def main():
    print("=== 大模型与 Function Call 交互演示 ===\n")

    # 测试不同情况
    test_cases = [
        "今天北京天气怎么样？",
        "计算一下 25+37*2 等于多少",
        "你好，我是新用户",
        "上海和广州的天气对比一下",
        "帮我算一下 (100-25)/3 的结果"
    ]

    for query in test_cases:
        chat_with_tools(query)
        print("-" * 50)


# 真实API调用示例（使用OpenAI）
def real_openai_example():
    """使用真实OpenAI API的示例"""

    # 注意：需要安装 openai 库并设置 API_KEY
    import os

    # 设置API密钥
    # os.environ["OPENAI_API_KEY"] = "your-api-key"

    # 创建OpenAI客户端
    client = openai.OpenAI()

    # 1. 第一次调用：让模型决定是否使用工具
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "今天北京天气怎么样？"}
        ],
        tools=fm.get_tools_schema(),  # 提供工具定义
        tool_choice="auto"  # 让模型自动决定
    )

    message = response.choices[0].message

    # 2. 如果模型调用了工具
    if message.tool_calls:
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # 执行工具
            result = fm.execute(func_name, arguments)

            # 3. 第二次调用：把工具结果给模型
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "今天北京天气怎么样？"},
                    message,  # 模型的工具调用请求
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }
                ]
            )

            print(f"最终回答: {second_response.choices[0].message.content}")


# 简化版的交互流程图
def visualize_interaction():
    """可视化交互流程"""
    print("\n" + "=" * 60)
    print("大模型与 Function Call 交互流程：")
    print("=" * 60)

    steps = [
        ("1. 用户输入", "👤: '今天北京天气怎么样？'"),
        ("2. 大模型分析", "🤖: '用户问天气，我需要调用天气工具'"),
        ("3. 返回工具调用", "📤: {'name': 'get_weather', 'arguments': {'city': '北京'}}"),
        ("4. 执行函数", "⚙️: 调用天气API获取数据"),
        ("5. 返回结果", "📥: '北京：晴天，25°C'"),
        ("6. 大模型生成回复", "🤖: '根据天气API，今天北京晴天，25°C...'"),
        ("7. 最终输出", "💬: '今天北京天气很好，晴天，温度25°C...'")
    ]

    for step, desc in steps:
        print(f"{step:20} {desc}")

    print("=" * 60)


# 多轮对话示例
def multi_turn_conversation():
    """多轮对话中的 Function Call"""

    llm = MockLLM()
    messages = [
        {"role": "system", "content": "你是有用的助手，可以调用工具。"}
    ]

    def process_round(user_input):
        messages.append({"role": "user", "content": user_input})

        # 第一次调用
        response = llm.chat_completion(messages, fm.get_tools_schema())
        message = response["choices"][0]["message"]

        if message.get("tool_calls"):
            # 处理工具调用
            messages.append(message)

            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                result = fm.execute(func_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result)
                })

            # 第二次调用获取最终回复
            final_response = llm.chat_completion(messages)
            assistant_msg = final_response["choices"][0]["message"]
            messages.append(assistant_msg)

            return assistant_msg["content"]
        else:
            messages.append(message)
            return message["content"]

    # 模拟对话
    print("\n=== 多轮对话示例 ===")

    queries = [
        "北京天气怎么样？",
        "那上海呢？",
        "计算一下两地的温差"
    ]

    for query in queries:
        print(f"\n👤: {query}")
        response = process_round(query)
        print(f"🤖: {response}")


if __name__ == "__main__":
    # 运行主演示
    main()

    # 显示交互流程
    visualize_interaction()

    # 运行多轮对话示例
    multi_turn_conversation()