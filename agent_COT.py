import asyncio

from agents import Agent, Runner

# Define an agent that always explains its reasoning step by step
cot_agent = Agent(
    name="ShoppingDiscountCoT",
    instructions=(
        "You are a shopping discount problem solver. "
        "Work out the solution step by step, then give the final answer."
    ),
)

# Example shopping discount question
question = (
    "A laptop costs 1299.99. " "It has a 15% discount. " "What is the final price?"
)

# Run the agent (using await in an async context, or Runner.run_sync in a script)
result = asyncio.run(Runner.run(cot_agent, input=question))
print(result.final_output)
