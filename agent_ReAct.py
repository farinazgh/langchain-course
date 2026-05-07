import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_product_price(product: str) -> float:
    print("get_product_price")
    """Look up the price of a product in the catalog."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@function_tool
def apply_discount(price: float, discount_percent: float) -> float:
    print("apply_discount")
    """Apply a discount percentage to a price and return the final price."""
    return round(price * (1 - discount_percent / 100), 2)


# Create an agent equipped with the tools
react_agent = Agent(
    name="ShoppingDiscountReAct",
    instructions=(
        "You are a shopping assistant. You have tools 'get_product_price' and 'apply_discount'. "
        "First, think step-by-step about the problem. If needed, use the tools to get prices and calculate discounts. "
        "After using a tool, reflect on the result and continue reasoning. "
        "After gathering information, provide the final answer."
    ),
    tools=[get_product_price, apply_discount],
)

# A shopping problem that requires using the tools
problem = (
    "I want to buy a laptop. "
    "First get the product price, then apply a 15% discount. "
    "What is the final price?"
)

result = asyncio.run(Runner.run(react_agent, input=problem))
print(result.final_output)
