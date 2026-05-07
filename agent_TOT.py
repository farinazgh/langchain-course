import asyncio

from agents import Agent, Runner


# Agent responsible for generating possible solution steps
generator_agent = Agent(
    name="ToT-Generator",
    instructions=(
        "Given the current situation, brainstorm a possible "
        "next step or action to reach the goal."
    ),
)

# Agent responsible for evaluating whether a proposed plan is useful
evaluator_agent = Agent(
    name="ToT-Evaluator",
    instructions=(
        "Assess how likely the proposed plan will solve the problem. "
        "Respond with either 'promising' or 'unlikely'."
    ),
)


problem_statement = """
You need to prepare for a technical certification exam in 4 weeks.
You can study theory, perform hands-on labs,
or take practice exams.
"""


async def main():

    print("\n======================================================")
    print("           TREE OF THOUGHTS (ToT) DEMO")
    print("======================================================\n")

    print("Problem Statement:")
    print(problem_statement)


    print("STEP 1 → Generate multiple candidate thoughts")
    print("These represent different possible starting strategies.\n")

    # ---------------------------------------------------------
    # STEP 1 — Generate multiple initial candidate thoughts
    # ---------------------------------------------------------

    initial_thoughts = []

    for branch_number in range(3):

        print("------------------------------------------------------")
        print(f"Generating candidate branch #{branch_number + 1}")
        print("------------------------------------------------------")

        generation_response = await Runner.run(
            generator_agent,
            input=(
                f"Problem:\n{problem_statement}\n\n"
                "Think of a good first step."
            ),
        )

        generated_thought = generation_response.final_output.strip()

        print("\nGenerated Thought:")
        print(f"→ {generated_thought}")

        print("\nThis thought becomes one branch")
        print("in the reasoning tree.\n")

        initial_thoughts.append(generated_thought)


    # ---------------------------------------------------------
    # STEP 2 — Evaluate each generated thought
    # ---------------------------------------------------------

    print("\n======================================================")
    print("STEP 2 → Evaluate each branch")
    print("======================================================\n")

    print("The evaluator agent will now judge whether")
    print("each branch looks useful or not.\n")

    promising_branches = []

    for branch_index, thought in enumerate(initial_thoughts, start=1):

        print("------------------------------------------------------")
        print(f"Evaluating branch #{branch_index}")
        print("------------------------------------------------------")

        print("\nCurrent branch content:")
        print(f"→ {thought}")

        evaluation_response = await Runner.run(
            evaluator_agent,
            input=f"Plan: {thought}\nIs this promising?",
        )

        evaluation_result = evaluation_response.final_output.strip()

        print("\nEvaluator Decision:")
        print(f"→ {evaluation_result}")

        # -----------------------------------------------------
        # STEP 3 — Expand only promising branches
        # -----------------------------------------------------

        if "promising" in evaluation_result.lower():

            print("\nResult:")
            print("This branch survived pruning.")
            print("The algorithm will now EXPAND this branch")
            print("by exploring the next reasoning step.\n")

            expansion_response = await Runner.run(
                generator_agent,
                input=(
                    f"Current idea: {thought}\n"
                    "Suggest the next logical step."
                ),
            )

            expanded_step = expansion_response.final_output.strip()

            full_branch = f"{thought} -> {expanded_step}"

            promising_branches.append(full_branch)

            print("Expanded Branch:")
            print(f"→ {full_branch}")

            print("\nThe reasoning tree has now grown deeper")
            print("for this promising path.\n")

        else:

            print("\nResult:")
            print("This branch was pruned.")
            print("The algorithm will stop exploring")
            print("this reasoning direction.\n")

        print("######################################################")
        print("Branch evaluation cycle completed.")
        print("The system has decided whether")
        print("to prune or expand this branch.")
        print("######################################################\n")

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n======================================================")
    print("FINAL SUMMARY")
    print("======================================================\n")

    print("Initial Candidate Thoughts:\n")

    for thought_index, thought in enumerate(initial_thoughts, start=1):
        print(f"{thought_index}. {thought}")

    print("\nPromising Expanded Branches:\n")

    if promising_branches:

        for branch_index, branch in enumerate(promising_branches, start=1):
            print(f"{branch_index}. {branch}")

    else:
        print("No promising branches were found.")

    print("\n======================================================")
    print("END OF TREE OF THOUGHTS DEMO")
    print("======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())