

作业：保险场景加功能
完成功能：①增加退保工具测试效果



"""
保险公司Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_insurance_products():
    """
    获取所有可用的保险产品列表
    """
    products = [
        {
            "id": "life_001",
            "name": "安心人寿保险",
            "type": "人寿保险",
            "description": "为您和家人提供长期保障",
            "min_amount": 100000,
            "max_amount": 5000000,
            "min_years": 10,
            "max_years": 30
        },
        {
            "id": "health_001",
            "name": "健康无忧医疗险",
            "type": "医疗保险",
            "description": "全面覆盖住院、门诊医疗费用",
            "min_amount": 50000,
            "max_amount": 1000000,
            "min_years": 1,
            "max_years": 5
        },
        {
            "id": "accident_001",
            "name": "意外伤害保险",
            "type": "意外险",
            "description": "保障意外伤害导致的医疗和伤残",
            "min_amount": 50000,
            "max_amount": 2000000,
            "min_years": 1,
            "max_years": 1
        }
    ]
    return json.dumps(products, ensure_ascii=False)


def get_product_detail(product_id: str):
    """
    获取指定保险产品的详细信息
    
    Args:
        product_id: 产品ID
    """
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
            "age_limit": "18-60周岁"
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
            "age_limit": "0-65周岁"
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
            "age_limit": "0-75周岁"
        }
    }
    
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


# ==================== 新增的退保工具函数 ====================
def calculate_surrender_value(product_id: str, insured_amount: int, years_paid: int, total_years: int):
    """
    计算退保价值（现金价值）

    Args:
        product_id: 产品ID
        insured_amount: 投保金额（元）
        years_paid: 已缴纳保费年数
        total_years: 总保障年限
    """
    # 退保价值计算逻辑（简化版）
    # 通常：现金价值 = 已缴保费总和 × 退保系数

    # 先获取产品详情以确定类型
    product_detail = json.loads(get_product_detail(product_id))

    if "error" in product_detail:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    product_type = product_detail.get("type", "")

    # 退保系数表（年限越长，退保损失越小）
    surrender_rates = {
        1: 0.3,  # 第1年退保：拿回30%
        2: 0.5,
        3: 0.6,
        4: 0.7,
        5: 0.8,
        10: 0.9,  # 第10年：拿回90%
        20: 1.0  # 第20年：拿回100%（假设为储蓄型产品）
    }

    # 找到最接近的退保系数
    surrender_rate = 0.2  # 默认20%
    for year, rate in sorted(surrender_rates.items()):
        if years_paid >= year:
            surrender_rate = rate
        else:
            break

    # 计算年保费（使用之前的calculate_premium函数）
    # 这里需要年龄参数，我们假设一个标准年龄35岁
    premium_result = json.loads(calculate_premium(product_id, insured_amount, total_years, 35))

    if "error" in premium_result:
        return json.dumps({"error": "无法计算保费"}, ensure_ascii=False)

    annual_premium = premium_result.get("annual_premium", 0)
    total_paid = annual_premium * years_paid

    # 计算退保价值
    surrender_value = total_paid * surrender_rate

    result = {
        "product_id": product_id,
        "product_name": product_detail.get("name", ""),
        "insured_amount": insured_amount,
        "years_paid": years_paid,
        "total_years": total_years,
        "total_premium_paid": round(total_paid, 2),
        "surrender_rate": f"{surrender_rate * 100}%",
        "surrender_value": round(surrender_value, 2),
        "loss_amount": round(total_paid - surrender_value, 2),
        "note": f"已缴纳{years_paid}年，共支付{round(total_paid, 2)}元，退保可拿回{round(surrender_value, 2)}元"
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
                        "description": "产品ID，例如：life_001, health_001, accident_001"
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
                "required": ["product_ids", "insured_amount", "years", "age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_surrender_value",
            "description": "计算保险产品的退保价值（现金价值），根据已缴纳年限和总年限计算能退回多少钱",
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
                    "years_paid": {
                        "type": "integer",
                        "description": "已缴纳保费的年数"
                    },
                    "total_years": {
                        "type": "integer",
                        "description": "总保障年限（年）"
                    }
                },
                "required": ["product_id", "insured_amount", "years_paid", "total_years"]
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
    "calculate_surrender_value": calculate_surrender_value
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
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的保险顾问助手。你可以：
            1. 介绍各种保险产品及其详细信息
            2. 根据客户需求计算保费
            3. 计算储蓄型保险的收益
            4. 比较不同保险产品
            请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
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
        
        response_message = response.choices[0].message
        
        # 将模型响应加入对话历史
        messages.append(response_message)
        
        # 检查是否需要调用工具
        # 型认为需要调用工具来回答问题，这里会是一个列表；如果模型可以直接回答，这里就是None。
        tool_calls = response_message.tool_calls
        
        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        # 在循环内，代码开始解读模型发来的“指令书”：
        for tool_call in tool_calls:
            function_name = tool_call.function.name# 1. 拿到“工具名称”
            function_args = json.loads(tool_call.function.arguments)# 2. 拿到“工具参数”
            
            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name] # 1. 根据名称找到对应的函数对象
                function_response = function_to_call(**function_args) # 2. 传入参数，执行函数
                
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
    # run_agent("你们有哪些保险产品？", model="qwen-plus")
    
    # 示例2：查询产品详情
    # run_agent("我想了解一下人寿保险的详细信息", model="qwen-plus")
    
    # 示例3：计算保费
    # run_agent("我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？", model="qwen-plus")
    
    # 示例4：计算收益
    # run_agent("如果我投保100万的人寿保险30年，到期能有多少收益？", model="qwen-plus")
    
    # 示例5：比较产品
    # run_agent("帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年", model="qwen-plus")

    # 测试退保问题
     run_agent("我买了50万人寿保险，已经交了3年保费，现在退保能拿回多少钱？", api_key='sk-xxx',model="qwen-plus")
    
    # 自定义查询
    #run_agent("你的问题", model="qwen-plus")


