# %% [markdown]
"""
# LLM Evaluation: Ensuring Quality and Reliability
**A Comprehensive Guide with Hands-On Examples**

*By Cristian Cordoba*

## What We'll Cover Today
1. **Foundations** - Basic metrics, validation, semantic similarity
2. **LLM-as-Judge** - Using Gemini to evaluate outputs
3. **RAG Evaluation** - Faithfulness, groundedness, citation accuracy
4. **Production** - CI/CD integration, monitoring, best practices
"""

# %% [markdown]
"""
## Part 1: Introduction & Foundations
"""

# %%
# Install dependencies (uncomment to run)
# !pip install -q pydantic langchain langchain-google-genai langchain-core sentence-transformers nltk numpy scipy scikit-learn rouge-score bert-score detoxify

# %%
# Imports
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import numpy as np
from typing import List, Optional

# Configuration
GEMINI_API_KEY = "AIzaSyB0nwBQqZ8Zxl_gOKFKZM8ZYx7EbNRAZG4"

# %% [markdown]
"""
## Why LLM Evaluation Matters

### Three Critical Functions:
1. **Ethical** 🤝 - Identify biases, ensure fairness
2. **Safety** 🛡️ - Prevent harmful content, reduce hallucinations
3. **Performance** ⚡ - Measure accuracy, optimize costs

### Business Case:
- **ROI**: 3-10x returns
- **Cost savings**: 20-40% reduction typical
- **User retention**: 15-25% increase with 10% quality improvement
"""

# %% [markdown]
"""
## LLM-Specific Risks

| Risk | Description | Impact |
|------|-------------|--------|
| **Hallucinations** 🎭 | False info presented as fact | Misinformation, legal liability |
| **Toxicity** ☠️ | Harmful/offensive content | Reputation damage, user harm |
| **Bias** ⚖️ | Systematic discrimination | Social inequity, legal violations |
| **Security** 🔓 | Prompt injection, jailbreaking | Data breaches, system compromise |

> **Key Insight**: These risks can be measured, monitored, and mitigated.
"""

# %%
# Example 1: Demonstrate Hallucination Risk
factual = "Paris is the capital of France."
hallucinated = "Paris is the capital of France, established in 1456 by King Louis XIV."

print("Factual:", factual)
print("Hallucinated:", hallucinated)
print("\nProblem: Simple string matching can't detect subtle hallucinations!")
print(f"Contains 'Paris'? Both: {('Paris' in factual) and ('Paris' in hallucinated)}")

# %% [markdown]
"""
## Traditional Testing vs LLM Evaluation

| Aspect | Traditional | LLM Evaluation |
|--------|------------|----------------|
| **Determinism** | Same input = same output | Probabilistic responses |
| **Correctness** | Binary pass/fail | Spectrum of quality |
| **Scope** | Function/module level | Task completion & UX |
| **Methods** | Automated assertions | Hybrid automated + human |
| **Timeline** | Pre-deployment | Continuous pre & post |

**Mindset Shift**: "Does it work?" → **"How well does it work?"**
"""

# %% [markdown]
"""
## Foundational Metrics: Precision, Recall, F1

- **Precision**: Of retrieved items, how many are relevant?
- **Recall**: Of relevant items, how many were retrieved?
- **F1**: Harmonic mean of precision and recall
"""

# %%
# Example 2: Implement Basic Metrics
def calculate_precision(retrieved: set, relevant: set) -> float:
    """Calculate precision for retrieval"""
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)

def calculate_recall(retrieved: set, relevant: set) -> float:
    """Calculate recall for retrieval"""
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Demo
retrieved_docs = {"doc1", "doc2", "doc3", "doc4"}
relevant_docs = {"doc2", "doc3", "doc5", "doc6"}

precision = calculate_precision(retrieved_docs, relevant_docs)
recall = calculate_recall(retrieved_docs, relevant_docs)
f1 = calculate_f1(precision, recall)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# %% [markdown]
"""
## Statistical Significance in A/B Testing

When comparing two models or prompts, we need statistical rigor to avoid false conclusions.

### Key Concepts:
- **P-value**: Probability the observed difference is due to chance (p < 0.05 = significant)
- **Confidence Interval**: Range of plausible values for the true difference  
  Example: 95% CI [0.02, 0.08] means we're 95% confident the true improvement is between 2-8%
- **Effect Size (Cohen's d)**: Magnitude of difference
  - 0.2 = small, 0.5 = medium, 0.8 = large
- **Bootstrap Resampling**: Empirical method to estimate confidence intervals without assumptions

### Example Interpretation:
```python
# Model A: mean=0.85, Model B: mean=0.88
# P-value: 0.03 (< 0.05) → Significant!
# 95% CI: [0.01, 0.05] → True difference likely 1-5%
# Cohen's d: 0.6 → Medium effect size
# Conclusion: Model B is statistically better
```

**Why This Matters**: Prevents deploying "improvements" that are just random noise.
"""

# %% [markdown]
"""
## Format Validation with Pydantic

Rule-based validation ensures outputs match expected schemas.
"""

# %%
# Example 3: Schema Validation
class ResponseFormat(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str]

# Valid response
try:
    valid = ResponseFormat(
        answer="Paris is the capital of France",
        confidence=0.95,
        sources=["Wikipedia", "Encyclopedia"]
    )
    print("✅ Valid response:", valid)
except ValidationError as e:
    print("❌ Validation failed:", e)

# Invalid response (confidence > 1.0)
try:
    invalid = ResponseFormat(
        answer="Paris",
        confidence=1.5,  # Invalid!
        sources=[]
    )
except ValidationError as e:
    print("\n❌ Invalid response caught:")
    print(f"   Confidence must be ≤ 1.0")

# %% [markdown]
"""
## Semantic Similarity Evolution

### Era 1: Lexical (BLEU, ROUGE)
- Word overlap counting
- Fast but surface-level

### Era 2: Neural (BERTScore)
- Embedding-based similarity
- Captures semantics

### Era 3: LLM-Assisted
- GPT-4/Gemini as judge
- Flexible, explainable
"""

# %%
# Example 4: Traditional NLP Metrics (BLEU)
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate1 = ["the", "cat", "sat", "on", "the", "mat"]  # Perfect match
candidate2 = ["a", "feline", "rested", "on", "the", "rug"]  # Semantic match

bleu1 = sentence_bleu(reference, candidate1)
bleu2 = sentence_bleu(reference, candidate2)

print(f"BLEU (exact match): {bleu1:.3f}")
print(f"BLEU (semantic match): {bleu2:.3f}")
print("\n⚠️ Limitation: BLEU can't recognize semantic equivalence!")

# %% [markdown]
"""
## ROUGE: Summarization Evaluation

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the industry standard for evaluating text summarization.

### Variants:
- **ROUGE-1**: Unigram (single word) overlap
- **ROUGE-2**: Bigram (two-word sequence) overlap  
- **ROUGE-L**: Longest Common Subsequence (captures sentence structure)

**Use Cases**: Summarization, text generation, paraphrasing
"""

# %%
# Example 4b: ROUGE for Summarization
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Reference summary
reference = "The quick brown fox jumps over the lazy dog in the garden."
# Generated summary
candidate = "A brown fox jumps over a dog in a garden."

scores = scorer.score(reference, candidate)

print("\n" + "=" * 60)
print("ROUGE SCORES")
print("=" * 60)
print(f"ROUGE-1 (unigram): {scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-2 (bigram): {scores['rouge2'].fmeasure:.3f}")
print(f"ROUGE-L (LCS): {scores['rougeL'].fmeasure:.3f}")
print("\n✅ Higher scores = better content overlap with reference")
print("=" * 60)

# %%
# Example 5: Embedding-Based Similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"

embeddings = model.encode([text1, text2])
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)

print(f"Semantic similarity: {similarity:.3f}")
print("✅ Embeddings capture semantic meaning beyond word overlap!")

# %% [markdown]
"""
## BERTScore: Contextual Semantic Similarity

BERTScore uses BERT embeddings to measure semantic similarity, going beyond simple word overlap.

### Advantages over BLEU/ROUGE:
- Captures **paraphrasing** and **synonyms**
- Understands **context** and **word order**
- Correlates better with **human judgment**

**Best for**: Evaluating when multiple valid phrasings exist
"""

# %%
# Example 5c: BERTScore - Contextual Semantic Similarity
from bert_score import score as bert_score

references = ["The cat sat on the mat"]
candidates = ["A feline rested on the rug"]

P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)

print("\n" + "=" * 60)
print("BERTSCORE vs BLEU Comparison")
print("=" * 60)
print(f"BERTScore F1: {F1.mean():.3f}")
print(f"BLEU Score: {bleu2:.3f}")
print("\n✅ BERTScore recognizes semantic equivalence!")
print("   'cat' ≈ 'feline', 'sat' ≈ 'rested'")
print("=" * 60)

# %% [markdown]
"""
## Inter-Rater Reliability: Validating LLM Judges

Before trusting an LLM-as-judge, validate it aligns with human judgment.

### Cohen's Kappa (κ) - Agreement Beyond Chance:
- **< 0.20**: Slight agreement
- **0.21-0.40**: Fair  
- **0.41-0.60**: Moderate ⚠️
- **0.61-0.80**: Substantial ✅
- **0.81-1.00**: Almost perfect ✅
"""

# %%
# Example 5b: Inter-Rater Reliability
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

def calculate_inter_rater_reliability(human_scores, llm_scores):
    """Measure agreement between human and LLM judge"""
    kappa = cohen_kappa_score(human_scores, llm_scores)
    agreement = np.mean(np.array(human_scores) == np.array(llm_scores))
    correlation, _ = pearsonr(human_scores, llm_scores)
    mae = np.mean(np.abs(np.array(human_scores) - np.array(llm_scores)))
    
    def interpret_kappa(k):
        if k < 0.20: return "Slight"
        elif k < 0.40: return "Fair"
        elif k < 0.60: return "Moderate"
        elif k < 0.80: return "Substantial"
        else: return "Almost perfect"
    
    return {
        'kappa': kappa,
        'interpretation': interpret_kappa(kappa),
        'agreement': agreement,
        'correlation': correlation,
        'mae': mae
    }

# Demo: Validate LLM judge against human ratings
print("\n" + "=" * 60)
print("INTER-RATER RELIABILITY: Human vs LLM Judge")
print("=" * 60)

# Simulated ratings (1-5 scale) for 20 responses
human_ratings = [5, 4, 3, 5, 4, 2, 5, 4, 3, 5, 4, 5, 3, 4, 2, 5, 4, 3, 5, 4]
llm_ratings =   [5, 4, 4, 5, 3, 2, 5, 4, 3, 4, 4, 5, 3, 4, 3, 5, 4, 3, 5, 3]

reliability = calculate_inter_rater_reliability(human_ratings, llm_ratings)

print(f"\nCohen's Kappa: {reliability['kappa']:.3f} ({reliability['interpretation']})")
print(f"Exact Agreement: {reliability['agreement']:.1%}")
print(f"Correlation: {reliability['correlation']:.3f}")
print(f"Mean Absolute Error: {reliability['mae']:.2f} points")

if reliability['kappa'] > 0.6:
    print("\n✅ SUBSTANTIAL agreement - LLM judge is reliable!")
elif reliability['kappa'] > 0.4:
    print("\n⚠️ MODERATE agreement - use with caution, validate critical cases")
else:
    print("\n❌ POOR agreement - needs calibration or different approach")
print("=" * 60)

# %% [markdown]
"""
## Pydantic for Structured Outputs

Ensures LLM evaluations return consistent, parseable results.
"""

# %%
# Example 6: Define Evaluation Schema
class EvaluationScore(BaseModel):
    """Structured evaluation result"""
    score: int = Field(ge=1, le=5, description="Score from 1 to 5")
    reasoning: str = Field(description="Explanation for the score")
    recommendation: str = Field(description="Pass or Fail")

# Example usage
eval_result = EvaluationScore(
    score=4,
    reasoning="Response is accurate and clear, but could be more detailed.",
    recommendation="Pass"
)
print(eval_result.model_dump_json(indent=2))

# %% [markdown]
"""
## Checkpoint: What We've Learned

✅ Basic metrics (Precision, Recall, F1)  
✅ Schema validation with Pydantic  
✅ Traditional NLP metrics (BLEU)  
✅ Embedding-based similarity  
✅ Structured output schemas  

**Next**: LLM-as-a-Judge with Gemini
"""

# %% [markdown]
"""
---
## Part 2: LLM-as-a-Judge

Using Gemini to evaluate LLM outputs programmatically.

### Key Benefits:
- **>80% agreement** with human evaluators
- **Reference-free** evaluation
- **Flexible criteria** via prompt engineering
- **Explainable** with reasoning
"""

# %%
# Example 7: Initialize Gemini Judge
def create_judge():
    """Create a Gemini model configured for structured evaluation"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.1,  # Low temperature for consistency
        google_api_key=GEMINI_API_KEY
    )
    return llm.with_structured_output(EvaluationScore)

judge = create_judge()
print("✅ Gemini judge initialized")

# %%
# Example 8: Evaluate a Good Response
task = "What is the capital of France?"
response = "Paris is the capital of France."

evaluation_prompt = f"""You are an expert evaluator. Score this LLM response on accuracy and clarity.

**Original Task:** {task}

**LLM Response:** {response}

**Instructions:**
- Score from 1-5 (1=poor, 5=excellent)
- Consider both factual accuracy and clarity
- Recommend "Pass" if score ≥ 4, otherwise "Fail"
- Provide clear reasoning
"""

result = judge.invoke([HumanMessage(content=evaluation_prompt)])

print("=" * 60)
print("EVALUATION RESULTS - Good Response")
print("=" * 60)
print(f"Score: {result.score}/5")
print(f"Reasoning: {result.reasoning}")
print(f"Recommendation: {result.recommendation}")
print("=" * 60)

# %%
# Example 9: Evaluate a Poor Response
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

print("\n" + "=" * 60)
print("EVALUATION RESULTS - Poor Response")
print("=" * 60)
print(f"Score: {result_2.score}/5")
print(f"Reasoning: {result_2.reasoning}")
print(f"Recommendation: {result_2.recommendation}")
print("=" * 60)

# %%
# Example 10: Multi-Criteria Evaluation
class MultiCriteriaScore(BaseModel):
    accuracy: int = Field(ge=1, le=5, description="Factual correctness")
    completeness: int = Field(ge=1, le=5, description="Covers all aspects")
    clarity: int = Field(ge=1, le=5, description="Easy to understand")
    overall_reasoning: str = Field(description="Overall assessment")

multi_judge = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
).with_structured_output(MultiCriteriaScore)

multi_prompt = """Evaluate this response on multiple criteria:

**Task:** Explain quantum computing
**Response:** Quantum computers use qubits that can be 0 and 1 simultaneously, enabling parallel processing for complex problems.

Score each criterion 1-5 and provide overall reasoning."""

multi_result = multi_judge.invoke([HumanMessage(content=multi_prompt)])
print("\nMulti-Criteria Evaluation:")
print(f"Accuracy: {multi_result.accuracy}/5")
print(f"Completeness: {multi_result.completeness}/5")
print(f"Clarity: {multi_result.clarity}/5")
print(f"Reasoning: {multi_result.overall_reasoning}")

# %%
# Example 11: Hallucination Detection
class HallucinationCheck(BaseModel):
    is_faithful: bool = Field(description="Is response faithful to context?")
    hallucinated_claims: List[str] = Field(description="List of unsupported claims")
    reasoning: str

hallucination_judge = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
).with_structured_output(HallucinationCheck)

context = "Our store is open Monday-Friday, 9 AM to 6 PM."
response_to_check = "We're open on weekends from 10 AM to 4 PM."

hallucination_prompt = f"""Check if this response is faithful to the context.

**Context:** {context}
**Response:** {response_to_check}

Identify any claims not supported by the context."""

hall_result = hallucination_judge.invoke([HumanMessage(content=hallucination_prompt)])
print("\nHallucination Detection:")
print(f"Faithful: {hall_result.is_faithful}")
print(f"Hallucinated claims: {hall_result.hallucinated_claims}")
print(f"Reasoning: {hall_result.reasoning}")

# %%
# Example 12: Batch Evaluation
def evaluate_response(task: str, response: str, criteria: str = "accuracy and clarity") -> EvaluationScore:
    """Helper function to evaluate a single response"""
    judge = create_judge()
    prompt = f"""Evaluate this LLM response on {criteria}.

**Task:** {task}
**Response:** {response}

Score 1-5. Recommend "Pass" (≥4) or "Fail" (<4)."""
    
    return judge.invoke([HumanMessage(content=prompt)])

test_cases = [
    {"task": "What is 2+2?", "response": "4", "criteria": "correctness"},
    {"task": "What is the meaning of life?", "response": "42", "criteria": "completeness and depth"},
    {"task": "Name three primary colors", "response": "Red, blue, and yellow", "criteria": "accuracy"}
]

print("\n" + "=" * 60)
print("BATCH EVALUATION RESULTS")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    result = evaluate_response(test["task"], test["response"], test["criteria"])
    print(f"\nTest {i}: {test['task']}")
    print(f"  Score: {result.score}/5 | {result.recommendation}")

# %% [markdown]
"""
## LLM-as-Judge Best Practices

### Prompt Engineering:
- ✅ Provide clear evaluation criteria
- ✅ Request step-by-step reasoning
- ✅ Use structured output formats
- ✅ Include examples when possible

### Bias Mitigation:
- ✅ Multiple evaluations for borderline cases
- ✅ Position swapping in pairwise comparisons
- ✅ Clear definitions to reduce ambiguity
- ✅ Regular calibration against human judgment
"""

# %% [markdown]
"""
## G-Eval: Reference-Free Evaluation with Chain-of-Thought

G-Eval is a state-of-the-art framework that uses LLM-as-judge with Chain-of-Thought reasoning.

### Key Features:
- **Reference-free**: No need for ground truth answers
- **Chain-of-Thought**: Step-by-step reasoning for transparency
- **Customizable criteria**: Evaluate any dimension (coherence, relevance, etc.)
- **High correlation**: >80% agreement with human evaluators

**When to use**: Complex, subjective quality assessment without reference texts
"""

# %%
# Example 12b: G-Eval Implementation
class GEval:
    """G-Eval: Reference-free evaluation with Chain-of-Thought"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def evaluate(self, task: str, output: str, criteria: str) -> dict:
        """
        Evaluate output using G-Eval methodology
        
        Args:
            task: Original task/prompt
            output: LLM output to evaluate
            criteria: What to evaluate (e.g., 'coherence', 'relevance')
        """
        eval_prompt = f"""You are an expert evaluator. Assess this output on {criteria}.

**Task:** {task}
**Output:** {output}

**Evaluation Steps:**
1. Analyze the output's {criteria}
2. Consider strengths and weaknesses
3. Provide a score from 1-5
4. Justify your reasoning

Score 1-5 where:
- 1 = Very poor {criteria}
- 3 = Acceptable {criteria}
- 5 = Excellent {criteria}

Provide detailed reasoning."""
        
        result = self.judge.invoke([HumanMessage(content=eval_prompt)])
        
        return {
            'criteria': criteria,
            'score': result.score,
            'reasoning': result.reasoning
        }

# Demo: G-Eval on multiple criteria
print("\n" + "=" * 60)
print("G-EVAL: MULTI-CRITERIA ASSESSMENT")
print("=" * 60)

g_eval = GEval(judge)

task = "Explain quantum computing to a beginner"
output = "Quantum computers use qubits that can be 0 and 1 simultaneously, enabling parallel processing for complex problems."

# Evaluate on multiple dimensions
criteria_list = ['coherence', 'relevance', 'clarity']

for criteria in criteria_list:
    result = g_eval.evaluate(task, output, criteria)
    print(f"\n{criteria.upper()}: {result['score']}/5")
    print(f"Reasoning: {result['reasoning'][:100]}...")

print("\n✅ G-Eval provides explainable, reference-free evaluation!")
print("=" * 60)

# %% [markdown]
"""
## G-Eval: Reference-Free Evaluation with Chain-of-Thought

G-Eval is a state-of-the-art framework that uses LLM-as-judge with Chain-of-Thought reasoning.

### Key Features:
- **Reference-free**: No need for ground truth answers
- **Chain-of-Thought**: Step-by-step reasoning for transparency
- **Customizable criteria**: Evaluate any dimension (coherence, relevance, etc.)
- **High correlation**: >80% agreement with human evaluators

**When to use**: Complex, subjective quality assessment without reference texts
"""

# %%
# Example 12b: G-Eval Implementation
class GEval:
    """G-Eval: Reference-free evaluation with Chain-of-Thought"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def evaluate(self, task: str, output: str, criteria: str) -> dict:
        """
        Evaluate output using G-Eval methodology
        
        Args:
            task: Original task/prompt
            output: LLM output to evaluate
            criteria: What to evaluate (e.g., 'coherence', 'relevance')
        """
        eval_prompt = f"""You are an expert evaluator. Assess this output on {criteria}.

**Task:** {task}
**Output:** {output}

**Evaluation Steps:**
1. Analyze the output's {criteria}
2. Consider strengths and weaknesses
3. Provide a score from 1-5
4. Justify your reasoning

Score 1-5 where:
- 1 = Very poor {criteria}
- 3 = Acceptable {criteria}
- 5 = Excellent {criteria}

Provide detailed reasoning."""
        
        result = self.judge.invoke([HumanMessage(content=eval_prompt)])
        
        return {
            'criteria': criteria,
            'score': result.score,
            'reasoning': result.reasoning
        }

# Demo: G-Eval on multiple criteria
print("\n" + "=" * 60)
print("G-EVAL: MULTI-CRITERIA ASSESSMENT")
print("=" * 60)

g_eval = GEval(judge)

task = "Explain quantum computing to a beginner"
output = "Quantum computers use qubits that can be 0 and 1 simultaneously, enabling parallel processing for complex problems."

# Evaluate on multiple dimensions
criteria_list = ['coherence', 'relevance', 'clarity']

for criteria in criteria_list:
    result = g_eval.evaluate(task, output, criteria)
    print(f"\n{criteria.upper()}: {result['score']}/5")
    print(f"Reasoning: {result['reasoning'][:100]}...")

print("\n✅ G-Eval provides explainable, reference-free evaluation!")
print("=" * 60)

# %% [markdown]
"""
## Task-Specific Evaluation Metrics

Different tasks require specialized metrics tailored to their objectives.
"""

# %%
# Example 13: Summarization Metrics
class SummarizationMetrics:
    """Comprehensive summarization evaluation"""
    
    @staticmethod
    def evaluate_summary(source: str, summary: str, reference: str = None) -> dict:
        """Evaluate summary quality across multiple dimensions"""
        metrics = {}
        
        # 1. Compression ratio
        source_len = len(source.split())
        summary_len = len(summary.split())
        metrics['compression_ratio'] = summary_len / source_len if source_len > 0 else 0
        
        # 2. Extractiveness (word overlap with source)
        source_words = set(source.lower().split())
        summary_words = set(summary.lower().split())
        if summary_words:
            metrics['extractiveness'] = len(summary_words & source_words) / len(summary_words)
        else:
            metrics['extractiveness'] = 0.0
        
        # 3. ROUGE (if reference available)
        if reference:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            metrics['rouge_l'] = scorer.score(reference, summary)['rougeL'].fmeasure
        
        return metrics

# Demo
print("\n" + "=" * 60)
print("SUMMARIZATION METRICS")
print("=" * 60)

source_text = "Artificial intelligence is transforming industries worldwide. Machine learning enables computers to learn from data without explicit programming. Deep learning uses neural networks with multiple layers for complex pattern recognition tasks."
summary = "AI and machine learning transform industries using neural networks for pattern recognition."
reference = "Artificial intelligence and machine learning are transforming industries through neural networks."

metrics = SummarizationMetrics.evaluate_summary(source_text, summary, reference)
print(f"Compression Ratio: {metrics['compression_ratio']:.2f}x")
print(f"Extractiveness: {metrics['extractiveness']:.1%}")
print(f"ROUGE-L vs Reference: {metrics.get('rouge_l', 'N/A'):.3f}")
print("=" * 60)

# %%
# Example 14: Question Answering Metrics
class QAMetrics:
    """Question answering evaluation metrics"""
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Exact match accuracy (case-insensitive)"""
        return prediction.strip().lower() == ground_truth.strip().lower()
    
    @staticmethod
    def f1_score_qa(prediction: str, ground_truth: str) -> float:
        """Token-level F1 score for QA"""
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        common = pred_tokens & truth_tokens
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

# Demo
print("\n" + "=" * 60)
print("QUESTION ANSWERING METRICS")
print("=" * 60)

question = "What is the capital of France?"
prediction = "Paris"
ground_truth = "Paris"

em = QAMetrics.exact_match(prediction, ground_truth)
f1 = QAMetrics.f1_score_qa(prediction, ground_truth)

print(f"Question: {question}")
print(f"Prediction: {prediction}")
print(f"Ground Truth: {ground_truth}")
print(f"\nExact Match: {em}")
print(f"F1 Score: {f1:.3f}")

# Test with partial match
prediction2 = "The capital is Paris, France"
f1_partial = QAMetrics.f1_score_qa(prediction2, ground_truth)
print(f"\nPartial answer F1: {f1_partial:.3f}")
print("=" * 60)

# %% [markdown]
"""
## Safety Evaluation: Toxicity Detection

Evaluating LLM outputs for harmful content is critical for production deployment.

### Key Safety Metrics:
- **Toxicity**: Overall harmful content score
- **Severe Toxicity**: Extremely harmful content
- **Obscenity**: Profane language
- **Threat**: Threatening language
- **Insult**: Insulting content
- **Identity Attack**: Attacks based on identity
"""

# %%
# Example 15: Toxicity Detection
from detoxify import Detoxify

def evaluate_toxicity(text: str) -> dict:
    """Detect toxic content in text"""
    model = Detoxify('original')
    results = model.predict(text)
    
    return {
        'toxicity': results['toxicity'],
        'severe_toxicity': results['severe_toxicity'],
        'obscene': results['obscene'],
        'threat': results['threat'],
        'insult': results['insult'],
        'identity_attack': results['identity_attack']
    }

# Demo
print("\n" + "=" * 60)
print("TOXICITY DETECTION")
print("=" * 60)

safe_text = "This is a helpful and respectful response to your question."
toxic_text = "You are stupid and worthless."

print("Safe Text Analysis:")
safe_scores = evaluate_toxicity(safe_text)
print(f"  Toxicity: {safe_scores['toxicity']:.3f}")
print(f"  Insult: {safe_scores['insult']:.3f}")

print("\nToxic Text Analysis:")
toxic_scores = evaluate_toxicity(toxic_text)
print(f"  Toxicity: {toxic_scores['toxicity']:.3f}")
print(f"  Insult: {toxic_scores['insult']:.3f}")

if toxic_scores['toxicity'] > 0.5:
    print("\n⚠️ HIGH TOXICITY DETECTED - Content should be filtered!")
else:
    print("\n✅ Content is safe")
print("=" * 60)

# %% [markdown]
"""
---
## Part 3: RAG Evaluation

### The 3-Layer Pyramid:
1. **Context Quality** (Foundation) - Retrieval precision/recall
2. **Robustness** (Middle) - Noise sensitivity, groundedness
3. **Response Quality** (Top) - Faithfulness, relevancy, citations
"""

# %%
# Example 15: Retrieval Evaluation
def evaluate_retrieval(retrieved: set, relevant: set) -> dict:
    """Evaluate retrieval quality"""
    precision = calculate_precision(retrieved, relevant)
    recall = calculate_recall(retrieved, relevant)
    f1 = calculate_f1(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}

# Demo
retrieved = {"doc1", "doc2", "doc3"}
relevant = {"doc2", "doc3", "doc4", "doc5"}

metrics = evaluate_retrieval(retrieved, relevant)
print("Retrieval Metrics:")
print(f"  Precision: {metrics['precision']:.2f}")
print(f"  Recall: {metrics['recall']:.2f}")
print(f"  F1: {metrics['f1']:.2f}")

# %% [markdown]
"""
## RAGAS Faithfulness: Industry-Standard RAG Evaluation

RAGAS (RAG Assessment) is the leading framework for evaluating RAG systems.

### Methodology:
1. **Extract** atomic statements from the generated response
2. **Verify** each statement against the retrieved context using LLM
3. **Calculate** faithfulness score = (supported statements) / (total statements)

**Score Interpretation**:
- 1.0 = Perfect faithfulness (no hallucinations)
- 0.8+ = Good (minor unsupported details)
- < 0.6 = Poor (significant hallucinations)
"""

# %%
# Example 16: RAGAS Faithfulness Implementation
class RAGASFaithfulness:
    """Implement RAGAS faithfulness metric"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate_faithfulness(self, context: str, response: str) -> dict:
        """Calculate faithfulness score using RAGAS methodology"""
        # Step 1: Extract atomic statements
        statements = self._extract_statements(response)
        
        if not statements:
            return {'faithfulness_score': 0.0, 'total_statements': 0, 'details': []}
        
        # Step 2: Verify each statement against context
        results = []
        for stmt in statements:
            supported = self._verify_statement(stmt, context)
            results.append({'statement': stmt, 'supported': supported})
        
        # Step 3: Calculate score
        supported_count = sum(1 for r in results if r['supported'])
        score = supported_count / len(statements)
        
        return {
            'faithfulness_score': score,
            'total_statements': len(statements),
            'supported_statements': supported_count,
            'details': results
        }
    
    def _extract_statements(self, response: str) -> List[str]:
        """Extract atomic statements from response"""
        statements = []
        for sent in response.replace('!', '.').replace('?', '.').split('.'):
            sent = sent.strip()
            if sent and len(sent) > 15:  # Filter short fragments
                statements.append(sent)
        return statements
    
    def _verify_statement(self, statement: str, context: str) -> bool:
        """Verify if statement can be inferred from context"""
        prompt = f"""Context: {context}

Statement: "{statement}"

Can this statement be directly inferred from the context? Answer YES or NO only."""
        
        try:
            result = self.judge.invoke([HumanMessage(content=prompt)])
            response_text = result.reasoning if hasattr(result, 'reasoning') else str(result)
            return 'yes' in response_text.lower()[:20]
        except:
            return False  # Conservative: assume not supported if verification fails

# Demo: RAGAS Faithfulness Evaluation
print("\n" + "=" * 60)
print("RAGAS FAITHFULNESS EVALUATION")
print("=" * 60)

rag_context = """Product: Premium Wireless Headphones
Price: $99
Shipping: 3-5 business days (standard shipping)
Free shipping on orders over $150
Battery: 30 hours"""

rag_response = "These headphones cost $99 with free shipping and have a 30-hour battery life."

# Initialize RAGAS evaluator
ragas_eval = RAGASFaithfulness(judge)
result = ragas_eval.calculate_faithfulness(rag_context, rag_response)

print(f"\nFaithfulness Score: {result['faithfulness_score']:.2f}")
print(f"Supported: {result['supported_statements']}/{result['total_statements']} statements")

print("\nDetailed Breakdown:")
for i, detail in enumerate(result['details'], 1):
    status = "✓ SUPPORTED" if detail['supported'] else "✗ UNSUPPORTED"
    print(f"  {i}. {status}: \"{detail['statement']}\"")

# Identify hallucinations
unsupported = [d['statement'] for d in result['details'] if not d['supported']]
if unsupported:
    print("\n⚠️ Potential Hallucinations:")
    for stmt in unsupported:
        print(f"  - {stmt}")
else:
    print("\n✅ No hallucinations detected - all claims grounded in context!")
print("=" * 60)

# %% [markdown]
"""
## Comprehensive RAG Metrics Suite

Beyond faithfulness, RAG systems require evaluation across multiple dimensions.
"""

# %%
# Example 17: Answer Relevancy
class AnswerRelevancy:
    """Measures how relevant the answer is to the question"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, answer: str) -> dict:
        """
        Evaluate if answer directly addresses the question
        Uses LLM to assess relevancy
        """
        prompt = f"""Evaluate if this answer is relevant to the question.

Question: {question}
Answer: {answer}

Score the relevancy from 1-5 where:
- 1 = Completely irrelevant
- 3 = Partially relevant
- 5 = Perfectly relevant and directly answers the question

Provide reasoning for your score."""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        
        return {
            'relevancy_score': result.score / 5.0,  # Normalize to 0-1
            'reasoning': result.reasoning
        }

# Demo
print("\n" + "=" * 60)
print("ANSWER RELEVANCY")
print("=" * 60)

ans_rel = AnswerRelevancy(judge)

question = "What are the health benefits of exercise?"
good_answer = "Exercise improves cardiovascular health, strengthens muscles, and boosts mental well-being."
irrelevant_answer = "The weather is nice today."

result_good = ans_rel.calculate(question, good_answer)
result_bad = ans_rel.calculate(question, irrelevant_answer)

print(f"Good Answer Relevancy: {result_good['relevancy_score']:.2f}")
print(f"Irrelevant Answer: {result_bad['relevancy_score']:.2f}")
print("=" * 60)

# %%
# Example 18: Context Precision
class ContextPrecision:
    """Measures if relevant contexts are ranked higher"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, contexts: List[str], ground_truth: str) -> dict:
        """
        Evaluate if relevant contexts appear first in retrieval results
        Higher precision = better ranking
        """
        relevance_scores = []
        
        for idx, context in enumerate(contexts):
            is_relevant = self._is_context_relevant(question, context, ground_truth)
            relevance_scores.append(is_relevant)
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for k, is_relevant in enumerate(relevance_scores, 1):
            if is_relevant:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precisions.append(precision_at_k)
        
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        
        return {
            'context_precision': avg_precision,
            'relevant_contexts': relevant_count,
            'total_contexts': len(contexts),
            'relevance_per_context': relevance_scores
        }
    
    def _is_context_relevant(self, question: str, context: str, ground_truth: str) -> bool:
        """Check if context is relevant for answering the question"""
        prompt = f"""Is this context relevant for answering the question?

Question: {question}
Context: {context}
Ground Truth Answer: {ground_truth}

Answer only: YES or NO"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return 'yes' in result.reasoning.lower()[:10]

# Demo
print("\n" + "=" * 60)
print("CONTEXT PRECISION")
print("=" * 60)

ctx_prec = ContextPrecision(judge)

question = "What is the capital of France?"
contexts = [
    "Paris is the capital and largest city of France.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is located in Paris."
]
ground_truth = "Paris"

result = ctx_prec.calculate(question, contexts, ground_truth)
print(f"Context Precision: {result['context_precision']:.2f}")
print(f"Relevant Contexts: {result['relevant_contexts']}/{result['total_contexts']}")
print("=" * 60)

# %%
# Example 19: Context Recall
class ContextRecall:
    """Measures if all necessary information is in retrieved contexts"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, contexts: List[str], ground_truth: str) -> dict:
        """
        Evaluate if retrieved contexts contain all information needed
        to answer the question correctly
        """
        # Extract key facts from ground truth
        facts = self._extract_facts(ground_truth)
        
        # Check which facts are covered by contexts
        combined_context = " ".join(contexts)
        covered_facts = []
        
        for fact in facts:
            is_covered = self._is_fact_in_context(fact, combined_context)
            covered_facts.append(is_covered)
        
        recall = sum(covered_facts) / len(covered_facts) if facts else 1.0
        
        return {
            'context_recall': recall,
            'covered_facts': sum(covered_facts),
            'total_facts': len(facts)
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text"""
        # Simplified: split by sentences
        facts = [s.strip() for s in text.split('.') if s.strip()]
        return facts
    
    def _is_fact_in_context(self, fact: str, context: str) -> bool:
        """Check if fact is present in context"""
        prompt = f"""Is this fact present in the context?

Fact: {fact}
Context: {context}

Answer only: YES or NO"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return 'yes' in result.reasoning.lower()[:10]

# Demo
print("\n" + "=" * 60)
print("CONTEXT RECALL")
print("=" * 60)

ctx_recall = ContextRecall(judge)

question = "What is the capital of France and its population?"
contexts = [
    "Paris is the capital of France.",
    "France is in Western Europe."
]
ground_truth = "Paris is the capital. Paris has a population of 2.2 million."

result = ctx_recall.calculate(question, contexts, ground_truth)
print(f"Context Recall: {result['context_recall']:.2f}")
print(f"Covered Facts: {result['covered_facts']}/{result['total_facts']}")
print("⚠️ Low recall means important information is missing from retrieval")
print("=" * 60)

# %%
# Example 20: Context Relevancy
class ContextRelevancy:
    """Measures how much of the retrieved context is actually relevant"""
    
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, context: str) -> dict:
        """
        Evaluate what percentage of context is relevant to the question
        Filters out noise and redundant information
        """
        # Split context into sentences
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        if not sentences:
            return {'context_relevancy': 0.0, 'relevant_sentences': 0, 'total_sentences': 0}
        
        # Check relevance of each sentence
        relevant_count = 0
        for sentence in sentences:
            if self._is_sentence_relevant(question, sentence):
                relevant_count += 1
        
        relevancy = relevant_count / len(sentences)
        
        return {
            'context_relevancy': relevancy,
            'relevant_sentences': relevant_count,
            'total_sentences': len(sentences)
        }
    
    def _is_sentence_relevant(self, question: str, sentence: str) -> bool:
        """Check if sentence is relevant to question"""
        prompt = f"""Is this sentence relevant for answering the question?

Question: {question}
Sentence: {sentence}

Answer only: YES or NO"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return 'yes' in result.reasoning.lower()[:10]

# Demo
print("\n" + "=" * 60)
print("CONTEXT RELEVANCY")
print("=" * 60)

ctx_relev = ContextRelevancy(judge)

question = "What is the capital of France?"
noisy_context = "Paris is the capital of France. The weather is nice. France has many cities. Paris is beautiful."

result = ctx_relev.calculate(question, noisy_context)
print(f"Context Relevancy: {result['context_relevancy']:.2f}")
print(f"Relevant Sentences: {result['relevant_sentences']}/{result['total_sentences']}")
print("✅ High relevancy = less noise in retrieved context")
print("=" * 60)

# %%
# Example 21: Answer Correctness
class AnswerCorrectness:
    """Combines factual correctness and semantic similarity"""
    
    def __init__(self, llm_judge, embedding_model):
        self.judge = llm_judge
        self.embedding_model = embedding_model
    
    def calculate(self, question: str, answer: str, ground_truth: str, 
                 factual_weight: float = 0.5) -> dict:
        """
        Evaluate answer correctness using both factual and semantic scores
        
        Args:
            factual_weight: Weight for factual score (0-1)
            semantic_weight = 1 - factual_weight
        """
        # Factual correctness (LLM-based)
        factual_score = self._evaluate_factual_correctness(question, answer, ground_truth)
        
        # Semantic similarity (embedding-based)
        semantic_score = self._calculate_semantic_similarity(answer, ground_truth)
        
        # Weighted combination
        semantic_weight = 1 - factual_weight
        overall_score = (factual_weight * factual_score + 
                        semantic_weight * semantic_score)
        
        return {
            'answer_correctness': overall_score,
            'factual_score': factual_score,
            'semantic_score': semantic_score,
            'weights': {'factual': factual_weight, 'semantic': semantic_weight}
        }
    
    def _evaluate_factual_correctness(self, question: str, answer: str, 
                                     ground_truth: str) -> float:
        """LLM-based factual correctness evaluation"""
        prompt = f"""Compare the answer with the ground truth for factual correctness.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Score factual correctness from 1-5 where:
- 1 = Completely incorrect
- 3 = Partially correct
- 5 = Completely correct

Provide reasoning."""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return result.score / 5.0  # Normalize to 0-1
    
    def _calculate_semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """Embedding-based semantic similarity"""
        embeddings = self.embedding_model.encode([answer, ground_truth])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

# Demo
print("\n" + "=" * 60)
print("ANSWER CORRECTNESS")
print("=" * 60)

ans_correct = AnswerCorrectness(judge, model)

question = "What is the capital of France?"
answer = "The capital city of France is Paris."
ground_truth = "Paris"

result = ans_correct.calculate(question, answer, ground_truth)
print(f"Answer Correctness: {result['answer_correctness']:.2f}")
print(f"  Factual Score: {result['factual_score']:.2f}")
print(f"  Semantic Score: {result['semantic_score']:.2f}")
print("=" * 60)

# %%
# Example 22: Answer Similarity
class AnswerSimilarity:
    """Semantic similarity between answer and reference"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def calculate(self, answer: str, reference: str) -> dict:
        """
        Calculate semantic similarity using embeddings
        More flexible than exact match
        """
        embeddings = self.embedding_model.encode([answer, reference])
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return {
            'answer_similarity': float(similarity),
            'interpretation': self._interpret_similarity(similarity)
        }
    
    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score"""
        if score > 0.9: return "Very similar"
        elif score > 0.7: return "Similar"
        elif score > 0.5: return "Somewhat similar"
        else: return "Different"

# Demo
print("\n" + "=" * 60)
print("ANSWER SIMILARITY")
print("=" * 60)

ans_sim = AnswerSimilarity(model)

answer = "Paris is the capital of France."
reference = "The capital city of France is Paris."

result = ans_sim.calculate(answer, reference)
print(f"Answer Similarity: {result['answer_similarity']:.3f}")
print(f"Interpretation: {result['interpretation']}")
print("✅ High similarity despite different wording!")
print("=" * 60)

# %%
# Example 23: Groundedness Check (Hallucination in RAG)
print("\n" + "=" * 60)
print("GROUNDEDNESS EXAMPLE")
print("=" * 60)
print(f"Context: {rag_context}")
print(f"Response: {rag_response}")
print(f"\n❌ Hallucination detected: 'free shipping' not in context!")
print(f"✅ Grounded claims: '$99', '3-5 days'")

# %%
# Example 18: Relevancy Evaluation
class RelevancyScore(BaseModel):
    relevancy_score: int = Field(ge=1, le=5)
    addresses_query: bool
    reasoning: str

relevancy_judge = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
).with_structured_output(RelevancyScore)

query = "What are the return policies?"
irrelevant_response = "We offer great customer service and support."

relevancy_prompt = f"""Evaluate if the response answers the query.

**Query:** {query}
**Response:** {irrelevant_response}

Score relevancy 1-5."""

rel_result = relevancy_judge.invoke([HumanMessage(content=relevancy_prompt)])
print("\nRelevancy Evaluation:")
print(f"Score: {rel_result.relevancy_score}/5")
print(f"Addresses query: {rel_result.addresses_query}")
print(f"Reasoning: {rel_result.reasoning}")

# %%
# Example 20: Complete RAG Evaluation Pipeline
def evaluate_rag_system(query: str, retrieved_docs: set, relevant_docs: set, 
                       context: str, response: str) -> dict:
    """End-to-end RAG evaluation"""
    metrics = {}
    
    # Layer 1: Context Quality
    metrics['retrieval'] = evaluate_retrieval(retrieved_docs, relevant_docs)
    
    # Layer 2 & 3: Use LLM-as-judge for faithfulness and relevancy
    # (In production, you'd call the judges here)
    
    metrics['summary'] = f"Precision: {metrics['retrieval']['precision']:.2f}, Recall: {metrics['retrieval']['recall']:.2f}"
    
    return metrics

# Demo
rag_metrics = evaluate_rag_system(
    query="Product pricing",
    retrieved_docs={"doc1", "doc2"},
    relevant_docs={"doc2", "doc3"},
    context=rag_context,
    response=rag_response
)

print("\n" + "=" * 60)
print("COMPLETE RAG EVALUATION")
print("=" * 60)
print(rag_metrics['summary'])

# %% [markdown]
"""
## RAG Failure Patterns

| Pattern | Detection | Mitigation |
|---------|-----------|------------|
| **Retrieval Failure** | Low recall + poor answers | Improve embeddings, query expansion |
| **Noise Sensitivity** | High precision, low groundedness | Better ranking, context filtering |
| **Hallucination** | Good retrieval, low faithfulness | Stronger prompts, different model |
| **Citation Errors** | High faithfulness, wrong sources | Improve citation instructions |
"""

# %% [markdown]
"""
---
## Part 4: Production & Best Practices
"""

# %%
# Example 22: CI/CD Evaluation Gate
def ci_cd_evaluation_gate(test_cases: list, min_score: float = 0.75) -> dict:
    """Automated evaluation pipeline for CI/CD"""
    print("Running CI/CD evaluation gate...")
    
    # Simulate batch evaluation
    scores = [4, 5, 3, 4, 5]  # Mock scores
    avg_score = sum(scores) / len(scores) / 5
    
    if avg_score < min_score:
        raise Exception(f"❌ Quality gate failed: {avg_score:.2f} < {min_score}")
    
    print(f"✅ Quality gate passed: {avg_score:.2f} ≥ {min_score}")
    return {"avg_score": avg_score, "passed": True}

# Demo
try:
    result = ci_cd_evaluation_gate(test_cases)
    print(f"Average score: {result['avg_score']:.2f}")
except Exception as e:
    print(e)

# %%
# Example 23: Production Safety Check
def production_safety_check(response: str) -> dict:
    """Real-time safety validation"""
    checks = {
        "length_ok": len(response) > 0 and len(response) < 5000,
        "no_pii": not any(pattern in response.lower() for pattern in ["ssn", "credit card"]),
        "format_valid": True  # Simplified
    }
    
    is_safe = all(checks.values())
    return {"safe": is_safe, "checks": checks}

# Demo
safety_result = production_safety_check("Paris is the capital of France.")
print("\nProduction Safety Check:")
print(f"Safe: {safety_result['safe']}")
print(f"Checks: {safety_result['checks']}")

# %%
# Example 24: Tiered Evaluation Strategy
class EvaluationTier:
    """Tiered evaluation for cost optimization"""
    
    @staticmethod
    def fast_check(response: str) -> bool:
        """100% of traffic - Fast validation"""
        return len(response) > 0 and len(response) < 5000
    
    @staticmethod
    def medium_check(response: str) -> bool:
        """10% sampling - Format validation"""
        # Simplified - would use Pydantic in production
        return True
    
    @staticmethod
    def comprehensive_check(response: str) -> dict:
        """1% sampling - Full LLM-as-judge evaluation"""
        # Would call actual judge in production
        return {"score": 4, "passed": True}

# Demo
import random
response = "Sample response"

# Fast check (100%)
if EvaluationTier.fast_check(response):
    print("✅ Fast check passed")
    
    # Medium check (10% sampling)
    if random.random() < 0.1:
        if EvaluationTier.medium_check(response):
            print("✅ Medium check passed (10% sample)")
    
    # Comprehensive check (1% sampling)
    if random.random() < 0.01:
        result = EvaluationTier.comprehensive_check(response)
        print(f"✅ Comprehensive check: {result}")

# %% [markdown]
"""
## Best Practices Checklist

### Before Deployment:
- [ ] Comprehensive test coverage (happy paths + edge cases)
- [ ] Clear metrics aligned with business goals
- [ ] Automated evaluation in CI/CD
- [ ] Human validation of automated metrics
- [ ] Safety checks (toxicity, bias, hallucinations)
- [ ] Security testing (prompt injection, jailbreaks)

### After Deployment:
- [ ] Continuous production monitoring
- [ ] Real-time safety checks
- [ ] Performance tracking (cost, latency, quality)
- [ ] User feedback collection
- [ ] Regular test set updates
- [ ] Quarterly calibration against human judgment
"""

# %% [markdown]
"""
## Conclusion & Key Takeaways

### 🎯 Core Principles:
1. **Multi-dimensional** - No single metric captures quality
2. **Continuous** - Evaluation spans entire lifecycle
3. **Hybrid approach** - Combine traditional, neural, and LLM methods
4. **Human alignment** - Automated metrics must match human judgment

### 📊 What We Covered:
✅ Foundational metrics (Precision, Recall, F1)  
✅ LLM-as-Judge with Gemini  
✅ RAG evaluation (Faithfulness, Groundedness)  
✅ Production deployment strategies  

### 💡 Next Steps:
1. Start with critical metrics for your use case
2. Implement LLM-as-judge for subjective quality
3. Build evaluation into CI/CD pipelines
4. Monitor continuously in production
5. Iterate based on real failures

### 🚀 Remember:
**Effective evaluation is the key to reliable LLM applications.**

---

*For more information, see the complete guide and documentation.*
"""

# %%
