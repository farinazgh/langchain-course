import re
import inspect
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

client = OpenAI()


# --- Tools: normal Python functions ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.


def get_tool_descriptions(tools_dict):
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers, if any.
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... this Thought/Action/Action Input/Observation can repeat N times
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# No tools= argument here.
# The LLM has no native tool-calling ability in this file.
# All agency comes from the prompt above and our regex parsing below.


@traceable(name="OpenAI Chat", run_type="llm")
def openai_chat_traced(messages, stop=None):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        stop=stop,
    )


# --- Agent Loop ---


@traceable(name="Raw ReAct Prompt Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad
        print(f"\n*** full_prompt {full_prompt} ***")

        # Stop token prevents the LLM from generating its own Observation.
        # We inject the real tool result instead.
        response = openai_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            stop=["\nObservation"],
        )

        output = response.choices[0].message.content or ""

        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        print(f"Final Answer:\n{final_answer_match}")

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # Parse tool calls from raw text with regex.
        # This is fragile if the LLM does not follow the format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format.
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools.keys())}"
            )
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # History is one growing string re-sent every iteration.
        # This replaces messages.append(...) from structured tool calling.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Raw ReAct Prompt Agent with OpenAI!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
