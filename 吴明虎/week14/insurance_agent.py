import os
import json

from openai import OpenAI


def get_doctor_list():
    """
    获取所有医生列表
    """
    doctors = [
        {
            "id": "doc_001",
            "name": "张三",
            "title": "主治医师",
            "experience": 10,
            "qualification": "医学博士",
            "specialty": "血液科",
            "registration_price": 100,
            "score": 4.5,
            "introduction": "华强医院，张三是主治医师，具有10年的经验"
        },
        {
            "id": "doc_002",
            "name": "李四",
            "title": "副主任医师",
            "experience": 5,
            "qualification": "医学硕士",
            "specialty": "外科",
            "registration_price": 80,
            "score": 4.2,
            "introduction": "华强医院，李四是副主任医师，具有5年的经验"
        },
        {
            "id": "doc_003",
            "name": "王五",
            "title": "主治医师",
            "experience": 8,
            "qualification": "医学博士",
            "specialty": "骨科",
            "registration_price": 120,
            "score": 4.7,
            "introduction": "华强医院，王五是主治医师，具有8年的经验"
        },
        {
            "id": "doc_004",
            "name": "李三",
            "title": "主治医师",
            "experience": 2,
            "qualification": "医学博士",
            "specialty": "血液科",
            "registration_price": 50,
            "score": 4.1,
            "introduction": "华强医院，张三是主治医师，具有2年的经验"
        },
        {
            "id": "doc_005",
            "name": "张四",
            "title": "副主任医师",
            "experience": 2,
            "qualification": "医学硕士",
            "specialty": "外科",
            "registration_price": 150,
            "score": 4.8,
            "introduction": "华强医院，李四是副主任医师，具有2年的经验"
        },
        {
            "id": "doc_006",
            "name": "赵五",
            "title": "主治医师",
            "experience": 5,
            "qualification": "医学博士",
            "specialty": "骨科",
            "registration_price": 100,
            "score": 4.1,
            "introduction": "华强医院，王五是主治医师，具有5年的经验"
        },

    ]
    return json.dumps(doctors, ensure_ascii=False)


def get_doctor_servers(doctor_id):
    servers = {
        "doc_001": {
            "id": "doc_001",
            "name": "普通检查",
            "type": "检查",
            "description": "检查血压，血脂,血蛋白检测",
            "coverage": ["血压", "血脂", "血蛋白"],
            "min_price": 150,
            "max_price": 500,
            "day_cus_num": 40,
        },
        "doc_004": {
            "id": "doc_004",
            "name": "普通检查",
            "type": "检查",
            "description": "检查血糖,血栓形成,血蛋白检测",
            "coverage": ["血糖", "血栓形成", "血蛋白"],
            "min_price": 300,
            "max_price": 2000,
            "day_cus_num": 10,
        },
        "doc_002": {
            "id": "doc_002",
            "name": "医师检查",
            "type": "检查",
            "description": "外伤诊断，上药，吊针",
            "coverage": ["外伤诊断", "上药", "吊针"],
            "min_price": 50,
            "max_price": 1500,
            "day_cus_num": 15,

        },
        "doc_005": {
            "id": "doc_005",
            "name": "医师检查",
            "type": "检查",
            "description": "烧伤，烫伤处理，上药，吊针",
            "coverage": ["烧伤", "烫伤", "上药", "吊针"],
            "min_price": 500,
            "max_price": 20000,
            "day_cus_num": 7,
        },
        "doc_003": {
            "id": "doc_003",
            "name": "医疗器械检查",
            "type": "检查",
            "description": "检查胸部X光，腹部X光,腿部X光，手部X光",
            "coverage": ["胸部X光", "腹部X光", "腿部X光", "手部X光"],
            "min_price": 1000,
            "max_price": 50000,
            "day_cus_num": 5,
        },
        "doc_006": {
            "id": "doc_006",
            "name": "医疗器械检查",
            "type": "检查",
            "description": "检查脑部X光，检查胸部X光，腹部X光,，bone扫描",
            "coverage": ["脑部X光", "bone扫描", "腹部X光", "手部X光"],
            "min_price": 15000,
            "max_price": 30000,
            "day_cus_num": 2,
        },
    }
    if doctor_id in servers:
        return json.dumps(servers[doctor_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "服务不存在"}, ensure_ascii=False)


#计算选择某个医生时需要付钱
def calculate_price(doctor_id, server_id, server_price_range: list, score, registration_price):
    score_weight = score - 4
    all_price = score_weight * (server_price_range[1] - server_price_range[0]) + registration_price
    result = {
        "doctor_id": doctor_id,
        "server_id": server_id,
        "all_mount": all_price,
        "calculation_note": f"基于{doctor_id}医生，服务{server_id}，总共需要花费{all_price}元"
    }

    return json.dumps(result, ensure_ascii=False)


#服务相同时选择性价比的医生
def compare_doctor_server(server_type: str, doctor_list: list, all_mount_list: list,score_list: list):
    value_doctor = doctor_list[0]
    v1=all_mount_list[0] * score_list[0]
    v2=all_mount_list[1] * score_list[1]
    value =  all_mount_list[0]
    score = score_list[0]
    if v1 > v2:
        value_doctor = doctor_list[1]
        value = all_mount_list[1]
        score = score_list[1]
    result = {
        "server_type": server_type,
        "value_doctor": value_doctor,
        "calculation_note": f"基于{server_type}服务，选择{value_doctor}医生最合适，价格为{value}，评分为{score}"
    }
    return json.dumps(result, ensure_ascii=False)

def find_month_best_value_doctor(doctor_list: list,day_cus_num_list:list, day_price_list: list):
    s= len(doctor_list)
    best=0
    best_value = 0
    for i in range(s):
        most_value = day_price_list[i] * day_cus_num_list[i]
        if most_value > best_value:
            best_value = most_value
            best = i
    result =  {
        "best_value_doctor": doctor_list[best],
        "best_value": best_value,
        "calculation_note": f"基于全体医生，选择{doctor_list[best]}医生作为本月医院最佳医生，服务价值为{best_value}"
    }
    return json.dumps(result, ensure_ascii=False)


tools = [
    {
        "type": "function",
        "function":     {
            "name": "get_doctor_list",
            "description": "获取所有医生列表",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
    },
    {
        "type": "function",
        "function":     {
            "name": "get_doctor_servers",
            "description": "获取指定医生的服务列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "医生ID，例如：doc_001，doc_002，doc_006"
                    }
                },
                "required": ["doctor_id"]
            }
        }
    },
    {
        "type": "function",
        "function":     {
            "name": "calculate_price",
            "description": "计算选择某个医生时需要付钱",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "医生ID，例如：doc_001，doc_002，doc_006"
                    },
                    "server_id": {
                        "type": "string",
                        "description": "服务ID"
                    },
                    "server_price_range": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "服务价格范围，两个参数为上下限"
                    },
                    "score": {
                        "type": "number",
                        "description": "医生评分，取值范围1-5"
                    },
                    "registration_price": {
                        "type": "number",
                        "description": "挂号价格"
                    },
                    "required": ["doctor_id", "server_id", "server_price_range"]

                }
            }
        }
    },
    {
        "type": "function",
        "function":     {
            "name": "compare_doctor_server",
            "description": "找到一个最合适的医生",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_type": {
                        "type": "string",
                        "description": "服务类型，例如：检查x光",
                    },
                    "doctor_list": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "能进行需要服务的医生列表",
                    },
                    "all_mount_list": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "医生列表对应需要服务的总价格列表",
                    },
                    "score_list": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "医生列表对应评分列表",
                    },
                    "required": ["server_type", "doctor_list", "all_mount_list"]
                }
            }
        }
    },
    {
        "type": "function",
        "function":     {
            "name": "find_month_best_value_doctor",
            "description": "寻找一个月以内医院最佳医生",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_list": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "进行评比的医生列表",
                    },
                    "day_cus_num_list": {
                        "type": "array",
                        "items": {},
                        "description": "医生列表对应每天服务人数列表",
                    },
                    "day_price_list": {
                        "type": "array",
                        "items": {},
                        "description": "医生列表对应每天服务价格列表",
                    }
                }
            }
        }
    }
]


available_functions = {
    "get_doctor_list": get_doctor_list,
    "get_doctor_servers": get_doctor_servers,
    "calculate_price": calculate_price,
    "compare_doctor_server": compare_doctor_server,
    "find_month_best_value_doctor": find_month_best_value_doctor,
}
from zai import ZhipuAiClient
from key_private import config

def run_agent(user_query: str, model: str = "qwen-plus"):
    """

    :param user_query:
    :param model:
    :return:
    """
    # client = ZhipuAiClient(api_key=config["chatGLM_api_key"])
    client = OpenAI(
        api_key=config["Aliyun_api_key"] or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "system",
            "content": """你是华强医院的一位专业的管理助手，你可以：
1. 介绍各个医生及其详细信息
2. 根据客户需求计算医疗费
3. 找到一个最合适的医生
4. 评选一个月最价值医生

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )
        # response = client.chat.completions.create(
        #     model="glm-4.7",
        #     messages=messages,
        #     temperature=0.6
        # )
        response_message = response.choices[0].message

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
        print("messages==================",messages)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:200]}..." if len(
                    function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


if __name__ == '__main__':
    # run_agent("请给我介绍一下医院的各个医生和他们的服务，不需要显示价格")
    # run_agent("我烫伤了，应该去找哪个医生看病，需要多少钱")
    run_agent("我需要进行血蛋白检测的检查，帮我找到一个最合适的医生")
    # run_agent("我需要进行血蛋白检测，帮我找一个医生，需要费用最低的")
    #run_agent("帮我寻找一个月以内医院最佳医生")
