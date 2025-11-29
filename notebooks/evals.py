# %% [markdown]
"""
# Simple LLM-as-Judge Example
# Using: Gemini + LangGraph + Pydantic

A minimal implementation showing how to evaluate LLM outputs programmatically.
"""

# %%
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# %% [markdown]
"""
## Configuration
Paste your Gemini API key below
"""

# %%
# TODO: Paste your Gemini API key here
GEMINI_API_KEY = "AIzaSyB0nwBQqZ8Zxl_gOKFKZM8ZYx7EbNRAZG4"

# %% [markdown]
"""
## Define Structured Output Schema
Using Pydantic to ensure consistent evaluation format
"""

# %%
class EvaluationScore(BaseModel):
    """Structured evaluation result"""
    score: int = Field(ge=1, le=5, description="Score from 1 to 5 (1=poor, 5=excellent)")
    reasoning: str = Field(description="Detailed explanation for the score")
    recommendation: str = Field(description="Pass or Fail")

# %% [markdown]
"""
## Initialize Gemini as Judge
Configure Gemini to output structured Pydantic objects
"""

# %%
def create_judge():
    """Create a Gemini model configured for structured evaluation"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,  # Low temperature for consistent evaluations
        google_api_key=GEMINI_API_KEY
    )
    # Bind Pydantic schema for structured output
    return llm.with_structured_output(EvaluationScore)

# %% [markdown]
"""
## Example 1: Evaluate a Simple Q&A Response
"""

# %%
# Initialize judge
judge = create_judge()

# Define what to evaluate
task = "What is the capital of France?"
response_to_evaluate = "Paris is the capital of France."

# Create evaluation prompt
evaluation_prompt = f"""You are an expert evaluator. Score this LLM response on accuracy and clarity.

**Original Task:** {task}

**LLM Response:** {response_to_evaluate}

**Instructions:**
- Score from 1-5 (1=poor, 5=excellent)
- Consider both factual accuracy and clarity
- Recommend "Pass" if score ≥ 4, otherwise "Fail"
- Provide clear reasoning
"""

# Get structured evaluation
result = judge.invoke([HumanMessage(content=evaluation_prompt)])

# Display results
print("=" * 60)
print("EVALUATION RESULTS - Example 1")
print("=" * 60)
print(f"Score: {result.score}/5")
print(f"Reasoning: {result.reasoning}")
print(f"Recommendation: {result.recommendation}")
print("=" * 60)

# %% [markdown]
"""
## Example 2: Evaluate a Poor Response
"""

# %%
task_2 = "Explain how photosynthesis works"
poor_response = "Plants make food from sun."

evaluation_prompt_2 = f"""You are an expert evaluator. Score this LLM response on completeness and depth.

**Original Task:** {task_2}

**LLM Response:** {poor_response}

**Instructions:**
- Score from 1-5 based on completeness, accuracy, and educational value
- Recommend "Pass" if score ≥ 4, otherwise "Fail"
- Explain what's missing or inadequate
"""

result_2 = judge.invoke([HumanMessage(content=evaluation_prompt_2)])

print(result_2)
print("\n" + "=" * 60)
print("EVALUATION RESULTS - Example 2")
print("=" * 60)
print(f"Score: {result_2.score}/5")
print(f"Reasoning: {result_2.reasoning}")
print(f"Recommendation: {result_2.recommendation}")
print("=" * 60)

# %% [markdown]
"""
## Example 3: Evaluate Multiple Responses (Batch)
"""

# %%
def evaluate_response(task: str, response: str, criteria: str = "accuracy and clarity") -> EvaluationScore:
    """Helper function to evaluate a single response"""
    judge = create_judge()
    
    prompt = f"""Evaluate this LLM response on {criteria}.

**Task:** {task}
**Response:** {response}

Score 1-5. Recommend "Pass" (≥4) or "Fail" (<4)."""
    
    return judge.invoke([HumanMessage(content=prompt)])


# Test cases
test_cases = [
    {
        "task": "What is 2+2?",
        "response": "2+2 equals 4",
        "criteria": "correctness"
    },
    {
        "task": "What is the meaning of life?",
        "response": "42",
        "criteria": "completeness and depth"
    },
    {
        "task": "Name three primary colors",
        "response": "Red, blue, and yellow are the three primary colors.",
        "criteria": "accuracy and completeness"
    }
]

print("\n" + "=" * 60)
print("BATCH EVALUATION RESULTS")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    result = evaluate_response(
        task=test["task"],
        response=test["response"],
        criteria=test["criteria"]
    )
    print(f"\nTest {i}: {test['task'][:50]}...")
    print(f"  Score: {result.score}/5")
    print(f"  Recommendation: {result.recommendation}")
    print(f"  Reasoning: {result.reasoning}")

# %% [markdown]
"""
## Key Features

✅ **Pydantic Structured Output** - Ensures consistent, parseable results  
✅ **Gemini Integration** - Uses Google's fast and cost-effective model  
✅ **Simple API** - Just 3 components: schema, judge, and prompt  
✅ **Reusable** - Easy to adapt for different evaluation criteria  

## Next Steps

1. Replace `GEMINI_API_KEY` with your actual key
2. Run the examples
3. Customize evaluation criteria for your use case
4. Add more sophisticated rubrics (multi-criteria scoring)
5. Integrate with LangGraph for workflow orchestration
"""

# %%
