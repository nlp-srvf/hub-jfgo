"""
    电商客服 Agent
    适配电商客服高频场景：查商品、查订单、算价格、处理售后、推荐商品
"""

import os
import json
from datetime import datetime, timedelta
from openai import OpenAI


# ==================== 模拟电商数据库 ====================
PRODUCTS_DB = {
    "prod_1001": {
        "id": "prod_1001",
        "name": "小米14 智能手机",
        "category": "手机",
        "price": 3999.00,
        "stock": 120,
        "specs": ["8GB+256GB", "12GB+256GB", "12GB+512GB"],
        "spec_prices": {"8GB+256GB": 3999, "12GB+256GB": 4299, "12GB+512GB": 4599},
        "discount": "满3000减200，支持6期免息",
        "after_sales": "7天无理由退货，1年全国联保"
    },
    "prod_1002": {
        "id": "prod_1002",
        "name": "华为Mate 60 Pro",
        "category": "手机",
        "price": 6999.00,
        "stock": 85,
        "specs": ["12GB+256GB", "12GB+512GB"],
        "spec_prices": {"12GB+256GB": 6999, "12GB+512GB": 7999},
        "discount": "无直降，赠送价值299元配件礼包",
        "after_sales": "7天无理由退货，2年全国联保"
    },
    "prod_2001": {
        "id": "prod_2001",
        "name": "Apple AirPods Pro 2",
        "category": "耳机",
        "price": 1799.00,
        "stock": 300,
        "specs": ["标准版", "USB-C版"],
        "spec_prices": {"标准版": 1799, "USB-C版": 1899},
        "discount": "满1500减100，支持3期免息",
        "after_sales": "14天无理由退货，1年官方保修"
    },
    "prod_3001": {
        "id": "prod_3001",
        "name": "小米空气净化器4 Pro",
        "category": "家电",
        "price": 1299.00,
        "stock": 58,
        "specs": ["标准版"],
        "spec_prices": {"标准版": 1299},
        "discount": "满1000减50，下单送滤芯",
        "after_sales": "7天无理由退货，2年质保"
    }
}

ORDERS_DB = {
    "order_8888": {
        "id": "order_8888",
        "user_id": "user_12345",
        "create_time": "2026-01-25 10:30:25",
        "status": "已发货",
        "products": [
            {"prod_id": "prod_1001", "spec": "12GB+256GB", "quantity": 1, "price": 4299.00},
            {"prod_id": "prod_2001", "spec": "USB-C版", "quantity": 1, "price": 1899.00}
        ],
        "total_amount": 6198.00,
        "paid_amount": 6198.00,
        "shipping_address": "北京市海淀区中关村大街1号",
        "logistics_no": "SF1234567890123",
        "logistics_company": "顺丰速运",
        "estimated_delivery": "2026-01-28"
    },
    "order_9999": {
        "id": "order_9999",
        "user_id": "user_12345",
        "create_time": "2026-01-27 15:45:10",
        "status": "待付款",
        "products": [
            {"prod_id": "prod_3001", "spec": "标准版", "quantity": 1, "price": 1299.00}
        ],
        "total_amount": 1299.00,
        "paid_amount": 0.00,
        "shipping_address": "上海市浦东新区张江高科技园区",
        "logistics_no": "",
        "logistics_company": "",
        "estimated_delivery": ""
    }
}

# 相似商品映射
SIMILAR_PRODUCTS = {
    "prod_1001": ["prod_1002"],
    "prod_1002": ["prod_1001"],
    "prod_2001": [],
    "prod_3001": []
}


# ==================== 工具函数定义 ====================
def get_product_info(prod_id: str = None, prod_name: str = None):
    """
    查询商品信息（支持按商品ID或商品名称查询）
    """
    result = {"products": []}

    # 按ID查询
    if prod_id and prod_id in PRODUCTS_DB:
        result["products"].append(PRODUCTS_DB[prod_id])
    # 按名称查询
    elif prod_name:
        for prod in PRODUCTS_DB.values():
            if prod_name.lower() in prod["name"].lower():
                result["products"].append(prod)
    # 无参数时返回所有商品
    else:
        result["products"] = list(PRODUCTS_DB.values())

    if not result["products"]:
        result["error"] = "未找到相关商品"
    else:
        result["count"] = len(result["products"])

    return json.dumps(result, ensure_ascii=False)


def get_order_status(order_id: str, user_id: str = None):
    """
    查询订单状态及详细信息
    """
    if order_id not in ORDERS_DB:
        return json.dumps({"error": f"订单ID {order_id} 不存在"}, ensure_ascii=False)

    order = ORDERS_DB[order_id]

    if user_id and order["user_id"] != user_id:
        return json.dumps({"error": "该订单不属于当前用户"}, ensure_ascii=False)

    status_explain = {
        "待付款": "订单已创建但未支付，需在24小时内完成付款",
        "已付款": "订单已支付，等待商家发货",
        "已发货": "订单已发货，可通过物流单号查询物流信息",
        "已完成": "订单已签收，交易完成",
        "已取消": "订单已取消，如已付款将原路退款",
        "售后中": "订单正在处理售后（退款/换货/维修）"
    }

    order["status_explain"] = status_explain.get(order["status"], "未知状态")

    return json.dumps(order, ensure_ascii=False)


def calculate_order_amount(prod_items: list, province: str = "北京"):
    """
    计算订单总价（含商品价格、优惠、运费）
    """

    # 基础运费规则
    freight_rules = {
        "包邮省份": ["北京", "上海", "广州", "深圳"],
        "基础运费": 10.00,
        "偏远地区运费": 20.00,
        "偏远省份": ["新疆", "西藏", "青海", "内蒙古"]
    }

    total_original = 0.00  # 原价总和
    total_discount = 0.00  # 优惠总和
    prod_details = []      # 商品明细

    # 计算商品总价和优惠
    for item in prod_items:
        prod_id = item.get("prod_id")
        spec = item.get("spec")
        quantity = item.get("quantity", 1)

        if prod_id not in PRODUCTS_DB:
            return json.dumps({"error": f"商品ID {prod_id} 不存在"}, ensure_ascii=False)

        product = PRODUCTS_DB[prod_id]
        spec_price = product["spec_prices"].get(spec, product["price"])

        # 计算单品原价
        item_original = spec_price * quantity
        total_original += item_original

        # 计算单品优惠（简化逻辑）
        item_discount = 0.00
        if "满3000减200" in product["discount"] and item_original >= 3000:
            item_discount = 200.00
        elif "满1500减100" in product["discount"] and item_original >= 1500:
            item_discount = 100.00
        elif "满1000减50" in product["discount"] and item_original >= 1000:
            item_discount = 50.00

        total_discount += item_discount

        prod_details.append({
            "prod_id": prod_id,
            "prod_name": product["name"],
            "spec": spec,
            "quantity": quantity,
            "spec_price": spec_price,
            "item_original": item_original,
            "item_discount": item_discount,
            "item_final": item_original - item_discount
        })

    # 计算运费
    if province in freight_rules["包邮省份"]:
        freight = 0.00
    elif province in freight_rules["偏远省份"]:
        freight = freight_rules["偏远地区运费"]
    else:
        freight = freight_rules["基础运费"]

    # 满88元包邮（通用规则）
    if (total_original - total_discount) >= 88:
        freight = 0.00

    # 最终总价
    final_amount = (total_original - total_discount) + freight

    result = {
        "prod_details": prod_details,
        "price_summary": {
            "total_original": round(total_original, 2),
            "total_discount": round(total_discount, 2),
            "freight": round(freight, 2),
            "final_amount": round(final_amount, 2)
        },
        "calculation_rules": {
            "discount_rules": "按商品优惠叠加计算",
            "freight_rules": f"包邮省份：{','.join(freight_rules['包邮省份'])}；偏远省份运费{freight_rules['偏远地区运费']}元；其他省份{freight_rules['基础运费']}元；满88元全国包邮"
        }
    }

    return json.dumps(result, ensure_ascii=False)


def handle_after_sales(order_id: str, after_sales_type: str, reason: str):
    """
    处理售后问题（退款/换货/投诉）
    """

    # 验证订单存在
    if order_id not in ORDERS_DB:
        return json.dumps({"error": f"订单ID {order_id} 不存在"}, ensure_ascii=False)

    order = ORDERS_DB[order_id]

    # 验证售后类型是否支持
    supported_types = ["退款", "换货", "投诉"]
    if after_sales_type not in supported_types:
        return json.dumps({"error": f"不支持的售后类型，仅支持：{','.join(supported_types)}"}, ensure_ascii=False)

    # 生成售后单号
    after_sales_id = f"as_{order_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 不同售后类型的处理逻辑
    process_rules = {
        "退款": {
            "conditions": ["商品未发货 或 商品已签收且符合7天无理由退货规则"],
            "process_time": "1-3个工作日",
            "refund_way": "原路返回至支付账户"
        },
        "换货": {
            "conditions": ["商品存在质量问题 或 商家发错货"],
            "process_time": "3-7个工作日",
            "note": "需先将商品寄回，运费由商家承担"
        },
        "投诉": {
            "conditions": ["无限制，所有订单均可投诉"],
            "process_time": "24小时内响应，3个工作日内给出处理结果",
            "note": "客服会尽快联系您核实情况"
        }
    }

    # 验证退款条件
    if after_sales_type == "退款":
        if order["status"] == "已发货" and order["estimated_delivery"] > datetime.now().strftime('%Y-%m-%d'):
            can_process = False
            process_note = "商品已发货且未签收，暂不支持退款，建议签收后申请7天无理由退货"
        else:
            can_process = True
            process_note = "符合退款条件，已为您创建退款申请"
    else:
        can_process = True
        process_note = f"已为您创建{after_sales_type}申请，客服会尽快处理"

    result = {
        "after_sales_id": after_sales_id,
        "order_id": order_id,
        "after_sales_type": after_sales_type,
        "reason": reason,
        "can_process": can_process,
        "process_note": process_note,
        "process_rules": process_rules[after_sales_type],
        "contact_way": "如有疑问可联系在线客服，或拨打400-123-4567"
    }

    return json.dumps(result, ensure_ascii=False)


def recommend_similar_products(prod_id: str):
    """
    推荐相似商品
    """
    if prod_id not in PRODUCTS_DB:
        return json.dumps({"error": f"商品ID {prod_id} 不存在"}, ensure_ascii=False)

    # 获取相似商品ID列表
    similar_prod_ids = SIMILAR_PRODUCTS.get(prod_id, [])
    similar_products = []

    for sp_id in similar_prod_ids:
        if sp_id in PRODUCTS_DB:
            similar_products.append(PRODUCTS_DB[sp_id])

    result = {
        "original_product": PRODUCTS_DB[prod_id],
        "similar_products": similar_products,
        "count": len(similar_products),
        "note": "为您推荐相似商品，您可对比选择" if similar_products else "暂无相似商品推荐"
    }

    return json.dumps(result, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "查询商品信息，包括价格、库存、规格、优惠政策、售后保障等，支持按商品ID精准查询或按商品名称模糊查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "prod_id": {
                        "type": "string",
                        "description": "商品ID，如：prod_1001, prod_2001"
                    },
                    "prod_name": {
                        "type": "string",
                        "description": "商品名称，用于模糊查询，如：小米14、AirPods Pro"
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "查询订单状态及详细信息，包括支付状态、物流信息、收货地址、预计送达时间等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单ID（必填），如：order_8888, order_9999"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "用户ID（可选），用于验证订单归属"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_order_amount",
            "description": "计算订单总价，包括商品原价、优惠金额、运费等，返回详细的价格明细",
            "parameters": {
                "type": "object",
                "properties": {
                    "prod_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prod_id": {"type": "string", "description": "商品ID"},
                                "spec": {"type": "string", "description": "商品规格"},
                                "quantity": {"type": "integer", "description": "购买数量，默认1"}
                            },
                            "required": ["prod_id", "spec"]
                        },
                        "description": "商品列表，包含ID、规格、数量"
                    },
                    "province": {
                        "type": "string",
                        "description": "收货省份，用于计算运费，如：北京、上海、广东"
                    }
                },
                "required": ["prod_items"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_after_sales",
            "description": "处理售后问题，包括退款申请、换货申请、投诉反馈，返回售后单号和处理规则",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单ID"
                    },
                    "after_sales_type": {
                        "type": "string",
                        "description": "售后类型，可选值：退款、换货、投诉"
                    },
                    "reason": {
                        "type": "string",
                        "description": "售后原因，如：商品质量问题、不想要了、发货错误等"
                    }
                },
                "required": ["order_id", "after_sales_type", "reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar_products",
            "description": "根据用户咨询的商品，推荐相似的替代商品，返回商品详细信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "prod_id": {
                        "type": "string",
                        "description": "商品ID"
                    }
                },
                "required": ["prod_id"],
                "additionalProperties": False
            }
        }
    }
]


# ==================== Agent核心逻辑 ====================
available_functions = {
    "get_product_info": get_product_info,
    "get_order_status": get_order_status,
    "calculate_order_amount": calculate_order_amount,
    "handle_after_sales": handle_after_sales,
    "recommend_similar_products": recommend_similar_products
}


def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行电商客服Agent，处理用户咨询
    """

    client = OpenAI(
        api_key="sk-6ad571eb032d4437a09bfb3ed5575591",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史（电商客服专属系统提示词）
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的电商客服助手，负责解答用户的各类购物咨询，包括：
                        1. 查询商品信息（价格、库存、规格、优惠、售后政策）
                        2. 查询订单状态（支付状态、物流信息、预计送达时间）
                        3. 计算订单价格（商品总价、优惠、运费）
                        4. 处理售后问题（退款、换货、投诉）
                        5. 推荐相似商品
                        
                        请根据用户的咨询内容，选择合适的工具获取准确信息，并以友好、专业的语气回复用户。
                        回复时要清晰明了，重点信息突出，符合电商客服的沟通规范。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "="*60)
    print("【用户咨询】")
    print(user_query)
    print("="*60)


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
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【客服回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False, indent=2)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:300]}..." if len(function_response) > 300 else f"工具返回: {function_response}")

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
    return "非常抱歉，暂时无法处理您的咨询，请稍后再试。"


# ==================== 示例 ====================
def demo_scenarios():
    """
    演示电商客服典型咨询场景
    """
    print("\n" + "#"*60)
    print("# 电商客服Agent演示 - Function Call能力展示")
    print("#"*60)

    scenarios = [
        "小米14手机多少钱？有什么规格？",
        "我的订单order_8888发货了吗？物流信息是多少？",
        "我想买1个小米14（12GB+256GB）和1个AirPods Pro 2（USB-C版），收货地是江苏省，总共需要多少钱？",
        "我要申请订单order_8888的退款，原因是不想要了",
        "推荐一些和小米14相似的手机"
    ]

    print("\n以下是电商客服典型咨询场景：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-"*60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-"*60)


if __name__ == "__main__":
    # 示例1：查询商品信息
    run_agent("小米14手机多少钱？有什么规格？", model="qwen-plus")

    # 示例2：查询订单状态
    # run_agent("我的订单order_8888发货了吗？物流信息是多少？", model="qwen-plus")

    # 示例3：计算订单价格
    # run_agent("我想买1个小米14（12GB+256GB）和1个AirPods Pro 2（USB-C版），收货地是江苏省，总共需要多少钱？", model="qwen-plus")

    # 示例4：处理售后
    # run_agent("我要申请订单order_8888的退款，原因是不想要了", model="qwen-plus")

    # 示例5：推荐相似商品
    # run_agent("推荐一些和小米14相似的手机", model="qwen-plus")