"""
保险公司Agent示例 - 大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from typing import Optional

from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义
products = {
    "life_001": {
        "id": "life_001",
        "name": "安心人寿保险",
        "type": "人寿保险",
        "description": "为您和家人提供长期保障，身故或全残可获赔付",
        "coverage": ["身故保障", "全残保障", "重大疾病保障"],
        "min_amount": 100000,
        "max_amount": 5000000,
        "min_years": 10,
        "max_years": 30,
        "age_limit": "18-60周岁",
        "wait_duration": "0天",
        "compensation_ratio": 1,
        "supporting_documents": ["受益人身份证明","保险合同号","死亡证明","关系证明"]
    },
    "health_001": {
        "id": "health_001",
        "name": "健康无忧医疗险",
        "type": "医疗保险",
        "description": "全面覆盖住院、门诊医疗费用",
        "coverage": ["住院医疗", "门诊医疗", "手术费用", "特殊门诊"],
        "min_amount": 50000,
        "max_amount": 1000000,
        "min_years": 1,
        "max_years": 5,
        "age_limit": "0-65周岁",
        "wait_duration": "60天",
        "compensation_ratio": 0.8,
        "supporting_documents": ["理赔申请表","医疗费用发票（原件）","诊断证明","出院小结（住院）"]
    },
    "accident_001": {
        "id": "accident_001",
        "name": "意外伤害保险",
        "type": "意外险",
        "description": "保障意外伤害导致的医疗和伤残",
        "coverage": ["意外身故", "意外伤残", "意外医疗"],
        "min_amount": 50000,
        "max_amount": 2000000,
        "min_years": 1,
        "max_years": 1,
        "age_limit": "0-75周岁",
        "wait_duration": "7天",
        "compensation_ratio": 0.9,
        "supporting_documents": ["理赔申请表","意外事故说明","诊断证明","医疗发票与费用清单"]
    }
}

# 此函数返回的信息已经在get_insurance_products中包含了，但是模型根据提示词的匹配度，仍然可能选择调用该函数。
# 但是，如果历史记录中已经调用过get_insurance_products获得了该信息，则不会再次使用该函数
def get_insurance_necessary_documents(product_id: str):
    """
    获取所需的理赔材料
    """
    if product_id in products:
        return json.dumps(products[product_id]['supporting_documents'], ensure_ascii=False)
    else:
        return json.dumps({
            'error': '产品不存在'
        }, ensure_ascii=False)

def get_insurance_products():
    """
    获取所有可用的保险产品列表
    """
    available_products = list(products.values())
    return json.dumps(available_products, ensure_ascii=False)


def get_product_detail(product_id: str):
    """
    获取指定保险产品的详细信息
    
    Args:
        product_id: 产品ID
    """

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


def calculate_premium(product_id: str, insured_amount: int, years: int, age: int):
    """
    计算保费
    
    Args:
        product_id: 产品ID
        insured_amount: 投保金额（元）
        years: 保障年限
        age: 投保人年龄
    """
    # 简化的保费计算逻辑（实际会更复杂）
    base_rates = {
        "life_001": 0.006,      # 人寿保险基础费率
        "health_001": 0.015,     # 医疗保险基础费率
        "accident_001": 0.002    # 意外险基础费率
    }
    
    if product_id not in base_rates:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)
    
    base_rate = base_rates[product_id]
    
    # 年龄系数（年龄越大，费率越高）
    age_factor = 1 + (age - 30) * 0.02 if age > 30 else 1
    
    # 年限系数
    year_factor = 1 + (years - 10) * 0.01 if years > 10 else 1
    
    # 计算年保费
    annual_premium = insured_amount * base_rate * age_factor * year_factor
    total_premium = annual_premium * years
    
    result = {
        "product_id": product_id,
        "insured_amount": insured_amount,
        "years": years,
        "age": age,
        "annual_premium": round(annual_premium, 2),
        "total_premium": round(total_premium, 2),
        "calculation_note": f"基于{age}岁投保，保障{years}年，保额{insured_amount}元"
    }
    
    return json.dumps(result, ensure_ascii=False)


def calculate_return(product_id: str, insured_amount: int, years: int):
    """
    计算保险收益（仅适用于有储蓄性质的保险）
    
    Args:
        product_id: 产品ID
        insured_amount: 投保金额（元）
        years: 保障年限
    """
    # 只有人寿保险有收益（储蓄型）
    if product_id == "life_001":
        # 假设年化收益率3.5%
        annual_rate = 0.035
        
        # 复利计算
        total_value = insured_amount * ((1 + annual_rate) ** years)
        total_return = total_value - insured_amount
        
        result = {
            "product_id": product_id,
            "insured_amount": insured_amount,
            "years": years,
            "annual_return_rate": f"{annual_rate * 100}%",
            "maturity_value": round(total_value, 2),
            "total_return": round(total_return, 2),
            "note": "此为储蓄型人寿保险的预期收益"
        }
        
        return json.dumps(result, ensure_ascii=False)
    else:
        return json.dumps({
            "error": "该产品为消费型保险，不提供收益计算",
            "note": "只有储蓄型保险产品才有收益"
        }, ensure_ascii=False)

# 这里虽然可以用于计算人寿险的赔数额（赔付比例1），但是会造成cost_amount表位可选项。
# 目前，大模型的tool call定义schema不完全支持条件参数，故不处理人寿险，以更清晰的告知大模型参数填充策略。
def calculate_compensation(product_id: str, cost_amount: int):
    """
    计算赔付金额

    Args:
        product_id: 产品ID
        cost_amount: 已花费金额（元）
    """
    # 只有医疗保险和意外险有赔付（储蓄型）
    if product_id == "life_001":
        result = {
            "error": "人寿险不支持赔付金额计算",
            "note": "此为寿险的赔付"
        }

        return json.dumps(result, ensure_ascii=False)
    elif product_id in products:
        product = products[product_id]
        compensation_amount = cost_amount * product['compensation_ratio']
        result = {
            "product_id": product_id,
            "compensation_amount": compensation_amount,
            "note": "此为报销型保险的赔付金额"
        }
        return json.dumps(result, ensure_ascii=False)
    else:
        return json.dumps({
            'error': '不存在此类保险'
        }, ensure_ascii=False)

def compare_products(product_ids: list, insured_amount: int, years: int, age: int):
    """
    比较多个保险产品的保费
    
    Args:
        product_ids: 产品ID列表
        insured_amount: 投保金额（元）
        years: 保障年限
        age: 投保人年龄
    """
    comparisons = []
    
    for product_id in product_ids:
        premium_result = json.loads(calculate_premium(product_id, insured_amount, years, age))
        if "error" not in premium_result:
            # 获取产品名称
            product_detail = json.loads(get_product_detail(product_id))
            premium_result["product_name"] = product_detail.get("name", "未知产品")
            comparisons.append(premium_result)
    
    # 按年保费排序
    comparisons.sort(key=lambda x: x["annual_premium"])
    
    result = {
        "comparison_params": {
            "insured_amount": insured_amount,
            "years": years,
            "age": age
        },
        "products": comparisons
    }
    
    return json.dumps(result, ensure_ascii=False)



# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_insurance_products",
            "description": "获取所有可用的保险产品列表，包括产品名称、类型、保额范围、年限范围等基本信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "获取指定保险产品的详细信息，包括保障范围、适用年龄等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID，例如：life_001, health_001, accident_002"  # 这里的示例可能导致大模型猜测产品ID，需要添加提示词来引导正确流程，而非使用禁止约束，以防止过度抑制
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_premium",
            "description": "计算指定保险产品的保费，包括年保费和总保费",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    },
                    "insured_amount": {
                        "type": "integer",
                        "description": "投保金额（元），即保额"
                    },
                    "years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    },
                    "age": {
                        "type": "integer",
                        "description": "投保人年龄"
                    }
                },
                "required": ["product_id", "insured_amount", "years", "age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_return",
            "description": "计算储蓄型保险产品到期后的收益，仅适用于有储蓄性质的保险（如人寿保险）",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    },
                    "insured_amount": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    }
                },
                "required": ["product_id", "insured_amount", "years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "比较多个保险产品在相同条件下的保费差异",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的产品ID列表"
                    },
                    "insured_amount": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    },
                    "age": {
                        "type": "integer",
                        "description": "投保人年龄"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_insurance_necessary_documents",   # 虽然返回的信息在产品列表中已经有了，但是在没有获取过产品列表的情况下，仍然会优先被选择使用。
            "description": "获得某个保险产品所需要的理赔材料",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_compensation',
            'description': '计算报销型保险的赔付金额',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_id': {
                        'type': 'string',
                        'description': '产品id'
                    },
                    'cost_amount': {
                        'type': 'integer',
                        'description': '花费金额，适用于报销型保险'
                    },
                    'required': ['product_id', 'cost_amount']
                }
            }
        }
    }
]



# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_insurance_products": get_insurance_products,
    "get_product_detail": get_product_detail,
    "calculate_premium": calculate_premium,
    "calculate_return": calculate_return,
    "compare_products": compare_products,
    "get_insurance_necessary_documents": get_insurance_necessary_documents,
    "calculate_compensation": calculate_compensation
}


def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询
    
    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 初始化对话历史
    # 使用一下prompt会导致模型不再自动使用tool call来推断产品ID
    # 1. 请不要使用示例的产品ID来推断用户提问中的产品ID
    # 2. 提示使用get_insurance_products获取产品列表再获取详细信息
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的保险顾问助手。你可以：
1. 介绍各种保险产品及其详细信息
2. 根据客户需求计算保费
3. 计算储蓄型保险的收益
4. 比较不同保险产品

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。如果客户的问题涉及到具体的某款产品，请先调用get_insurance_products列出产品列表，然后使用产品中的ID查询其他信息。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)
    
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
        print(response)
        response_message = response.choices[0].message
        
        # 将模型响应加入对话历史
        messages.append(response_message)
        
        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls
        
        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")
                
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




# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#"*60)
    print("# 保险公司Agent演示 - Function Call能力展示")
    print("#"*60)
    
    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数
    
    scenarios = [
        "你们有哪些保险产品？",
        "我想了解一下人寿保险的详细信息",
        "我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？",
        "如果我投保100万的人寿保险30年，到期能有多少收益？",
        "帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年"
    ]
    
    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    print("\n" + "-"*60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-"*60)



if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()
    
    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY
    
    # 示例1：查询产品列表
    # run_agent("你们有哪些保险产品？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model="qwen-plus")
    
    # 示例2：查询产品详情，这里prompt保证了需要多步思考来达成目的: 1. 获得产品列表 2. 根据产品名获得产品Id 3. 根据产品Id获得具体的产品信息
    # run_agent("我想了解一下人寿保险的详细信息", api_key='sk-e87d5ce46d8994113afb179546c459f81', model="qwen-plus")
    
    # 示例3：计算保费
    # run_agent("我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model="qwen-plus")
    
    # 示例4：计算收益
    # run_agent("如果我投保100万的人寿保险30年，到期能有多少收益？", model="qwen-plus")
    
    # 示例5：比较产品
    # run_agent("帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年", model="qwen-plus")
    
    # 自定义查询，查询需要的理赔材料，需要prompt来暗示范围，比如你们、产品编号、所属公司。否则容易生成泛华的回复
    # 原tool call的描述中存在"产品ID，例如：life_001, health_001, accident_001"，可能出现意外伤害险=>accident_001作为ID的轻度幻觉。
    # run_agent("我购买了你们的意外伤害险，需要提供什么材料索赔？", api_key='sk-e87d5ce4d8994113afb179546c459f81', model="qwen-plus")

    # 自定义查询，报销型保险的赔付金额
    # run_agent("如果我有100000元的意外伤害险的保额，我的医疗花费是10000元，我可以获得多少赔偿？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model="qwen-plus")

    # 自定义查询，人寿险的赔付金额
    run_agent("如果我有100000元的人寿险，我的医疗花费是10000元，我可以获得多少赔偿？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model="qwen-plus")




