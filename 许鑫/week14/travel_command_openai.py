from openai import OpenAI
import json

client = OpenAI()

# 模拟数据库
TRAVEL_DB = {
    "台北": ["台北101", "故宫博物院", "士林夜市"],
    "东京": ["东京塔", "浅草寺", "秋叶原"],
    "巴黎": ["埃菲尔铁塔", "卢浮宫", "凯旋门"]
}


# 真实函数
def get_travel_spots(city: str):
    spots = TRAVEL_DB.get(city)
    if not spots:
        return {"error": "未找到该城市数据"}
    return {"city": city, "spots": spots}


# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_travel_spots",
            "description": "根据城市名称推荐旅游景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

# 用户输入
user_input = "推荐东京的旅游景点"

# 第一步：让模型决定是否调用函数
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "当用户询问旅游景点时必须调用函数，不要直接回答。"},
        {"role": "user", "content": user_input}
    ],
    tools=tools
)

message = response.choices[0].message

# 第二步：判断是否触发 function call
if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    # 第三步：路由执行函数
    if function_name == "get_travel_spots":
        result = get_travel_spots(**arguments)

    # 第四步：把函数结果回传给模型
    second_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "请根据函数返回结果生成自然语言推荐"},
            {"role": "user", "content": user_input},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]
    )

    print(second_response.choices[0].message.content)

else:
    print(message.content)
