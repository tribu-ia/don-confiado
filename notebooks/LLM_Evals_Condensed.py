# %% [markdown]
"""
# LLM Evaluation: Essential Concepts
**A Focused Guide with Key Examples**

*By Cristian Cordoba*

## What We'll Cover
1. **Foundations** - Core metrics and validation
2. **LLM-as-Judge** - Automated evaluation with Gemini
3. **RAG Evaluation** - Retrieval and generation quality
4. **Production** - Deployment best practices
"""

# %% [markdown]
"""
## Part 1: Foundations
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

### Key Risks:
- **Hallucinations** 🎭 - False information
- **Toxicity** ☠️ - Harmful content
- **Bias** ⚖️ - Discrimination
- **Security** 🔓 - Prompt injection

### Business Impact:
- ROI: 3-10x returns
- Cost savings: 20-40%
- User retention: +15-25%
"""

# %% [markdown]
"""
## Example 1: Basic Metrics (Precision, Recall, F1)

Foundation for all retrieval evaluation.
"""

# %%
def calculate_precision(retrieved: set, relevant: set) -> float:
    if not retrieved: return 0.0
    return len(retrieved & relevant) / len(retrieved)

def calculate_recall(retrieved: set, relevant: set) -> float:
    if not relevant: return 0.0
    return len(retrieved & relevant) / len(relevant)

def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Demo
retrieved = {"doc1", "doc2", "doc3", "doc4"}
relevant = {"doc2", "doc3", "doc5", "doc6"}

p = calculate_precision(retrieved, relevant)
r = calculate_recall(retrieved, relevant)
f1 = calculate_f1(p, r)

print(f"Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")

# %% [markdown]
"""
## Statistical Significance in A/B Testing

**Key Concepts** (conceptual understanding):
- **P-value**: p < 0.05 = significant difference
- **Confidence Interval**: 95% CI shows plausible range
- **Effect Size (Cohen's d)**: 0.2=small, 0.5=medium, 0.8=large

**Why**: Prevents deploying "improvements" that are random noise.
"""

# %% [markdown]
"""
## Example 2: BLEU Score (Traditional NLP)

N-gram overlap metric for text generation.
"""

# %%
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate1 = ["the", "cat", "sat", "on", "the", "mat"]  # Perfect
candidate2 = ["a", "feline", "rested", "on", "the", "rug"]  # Semantic

bleu1 = sentence_bleu(reference, candidate1)
bleu2 = sentence_bleu(reference, candidate2)

print(f"BLEU (exact): {bleu1:.3f} | BLEU (semantic): {bleu2:.3f}")
print("⚠️ Limitation: Can't recognize semantic equivalence")

# %% [markdown]
"""
## Example 3: ROUGE Score (Summarization Standard)

Industry standard for evaluating summaries.
"""

# %%
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The quick brown fox jumps over the lazy dog in the garden."
candidate = "A brown fox jumps over a dog in a garden."

scores = scorer.score(reference, candidate)

print("ROUGE Scores:")
print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
print(f"  ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
print(f"  ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

# %% [markdown]
"""
## Example 4: BERTScore (Semantic Similarity)

Embedding-based metric that captures semantic meaning.
"""

# %%
from bert_score import score as bert_score

references = ["The cat sat on the mat"]
candidates = ["A feline rested on the rug"]

P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)

print(f"BERTScore F1: {F1.mean():.3f} vs BLEU: {bleu2:.3f}")
print("✅ BERTScore recognizes semantic equivalence!")

# %% [markdown]
"""
## Example 5: Inter-Rater Reliability (Validation)

Validate LLM judges against human judgment using Cohen's Kappa.
"""

# %%
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

def calculate_reliability(human_scores, llm_scores):
    kappa = cohen_kappa_score(human_scores, llm_scores)
    agreement = np.mean(np.array(human_scores) == np.array(llm_scores))
    correlation, _ = pearsonr(human_scores, llm_scores)
    
    interpretation = (
        "Almost perfect" if kappa > 0.8 else
        "Substantial" if kappa > 0.6 else
        "Moderate" if kappa > 0.4 else "Fair"
    )
    
    return {'kappa': kappa, 'interpretation': interpretation, 
            'agreement': agreement, 'correlation': correlation}

# Demo
human = [5, 4, 3, 5, 4, 2, 5, 4, 3, 5]
llm =   [5, 4, 4, 5, 3, 2, 5, 4, 3, 4]

rel = calculate_reliability(human, llm)
print(f"Cohen's Kappa: {rel['kappa']:.3f} ({rel['interpretation']})")
print(f"Agreement: {rel['agreement']:.1%} | Correlation: {rel['correlation']:.3f}")

# %% [markdown]
"""
## Example 6: Pydantic for Structured Outputs

Ensures consistent, parseable evaluation results.
"""

# %%
class EvaluationScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Score 1-5")
    reasoning: str = Field(description="Explanation")
    recommendation: str = Field(description="Pass or Fail")

# Demo
eval_result = EvaluationScore(
    score=4,
    reasoning="Accurate and clear, but could be more detailed.",
    recommendation="Pass"
)
print(eval_result.model_dump_json(indent=2))

# %% [markdown]
"""
---
## Part 2: LLM-as-a-Judge

Using Gemini to evaluate LLM outputs with >80% human agreement.
"""

# %% [markdown]
"""
## Example 7: Initialize Gemini Judge
"""

# %%
def create_judge():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.1,
        google_api_key=GEMINI_API_KEY
    )
    return llm.with_structured_output(EvaluationScore)

judge = create_judge()
print("✅ Gemini judge initialized")

# %% [markdown]
"""
## Example 8: Simple Evaluation (Good vs Poor)

Compare quality across different responses.
"""

# %%
def evaluate_response(task: str, response: str) -> EvaluationScore:
    prompt = f"""Evaluate this response on accuracy and clarity.

Task: {task}
Response: {response}

Score 1-5. Recommend "Pass" (≥4) or "Fail" (<4)."""
    
    return judge.invoke([HumanMessage(content=prompt)])

# Demo: Good vs Poor
task = "Explain photosynthesis"
good = "Photosynthesis converts light energy into chemical energy using chlorophyll in plants."
poor = "Plants make food from sun."

result_good = evaluate_response(task, good)
result_poor = evaluate_response(task, poor)

print("Good Response:")
print(f"  Score: {result_good.score}/5 | {result_good.recommendation}")

print("\nPoor Response:")
print(f"  Score: {result_poor.score}/5 | {result_poor.recommendation}")
print(f"  Issue: {result_poor.reasoning[:80]}...")

# %% [markdown]
"""
## Example 9: Multi-Criteria Evaluation

Assess multiple dimensions simultaneously.
"""

# %%
class MultiCriteriaScore(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall: str

multi_judge = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
).with_structured_output(MultiCriteriaScore)

prompt = """Evaluate on accuracy, completeness, and clarity (1-5 each):

Task: Explain quantum computing
Response: Quantum computers use qubits that can be 0 and 1 simultaneously."""

result = multi_judge.invoke([HumanMessage(content=prompt)])
print(f"Accuracy: {result.accuracy}/5")
print(f"Completeness: {result.completeness}/5")
print(f"Clarity: {result.clarity}/5")

# %% [markdown]
"""
## Example 10: G-Eval (Reference-Free with CoT)

State-of-the-art evaluation with Chain-of-Thought reasoning.
"""

# %%
class GEval:
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def evaluate(self, task: str, output: str, criteria: str) -> dict:
        prompt = f"""Assess this output on {criteria}.

Task: {task}
Output: {output}

Steps:
1. Analyze {criteria}
2. Consider strengths/weaknesses
3. Score 1-5
4. Justify reasoning"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return {'criteria': criteria, 'score': result.score, 'reasoning': result.reasoning}

# Demo
g_eval = GEval(judge)
task = "Explain quantum computing to a beginner"
output = "Quantum computers use qubits for parallel processing."

for criteria in ['coherence', 'relevance']:
    result = g_eval.evaluate(task, output, criteria)
    print(f"{criteria.upper()}: {result['score']}/5")

# %% [markdown]
"""
## Example 11: Toxicity Detection (Safety)

Critical for production deployment.
"""

# %%
from detoxify import Detoxify

def evaluate_toxicity(text: str) -> dict:
    model = Detoxify('original')
    results = model.predict(text)
    return {
        'toxicity': results['toxicity'],
        'insult': results['insult'],
        'threat': results['threat']
    }

# Demo
safe = "This is a helpful response."
toxic = "You are stupid and worthless."

safe_scores = evaluate_toxicity(safe)
toxic_scores = evaluate_toxicity(toxic)

print(f"Safe text toxicity: {safe_scores['toxicity']:.3f}")
print(f"Toxic text toxicity: {toxic_scores['toxicity']:.3f}")

if toxic_scores['toxicity'] > 0.5:
    print("⚠️ HIGH TOXICITY - Filter this content!")

# %% [markdown]
"""
**LLM-as-Judge Best Practices:**
- ✅ Clear evaluation criteria
- ✅ Step-by-step reasoning
- ✅ Structured outputs
- ✅ Regular calibration vs humans
"""

# %% [markdown]
"""
---
## Part 3: RAG Evaluation

Evaluating retrieval and generation quality.
"""

# %% [markdown]
"""
## Example 12: Retrieval Metrics (P/R/F1)

Foundation for RAG evaluation.
"""

# %%
retrieved = {"doc1", "doc2", "doc3"}
relevant = {"doc2", "doc3", "doc4", "doc5"}

metrics = {
    'precision': calculate_precision(retrieved, relevant),
    'recall': calculate_recall(retrieved, relevant)
}
metrics['f1'] = calculate_f1(metrics['precision'], metrics['recall'])

print(f"Retrieval - P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}")

# %% [markdown]
"""
## Example 13: RAGAS Faithfulness

Industry-standard RAG evaluation - detects hallucinations.
"""

# %%
class RAGASFaithfulness:
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, context: str, response: str) -> dict:
        statements = self._extract_statements(response)
        if not statements:
            return {'score': 0.0, 'details': []}
        
        results = []
        for stmt in statements:
            supported = self._verify(stmt, context)
            results.append({'statement': stmt, 'supported': supported})
        
        score = sum(1 for r in results if r['supported']) / len(statements)
        return {'faithfulness_score': score, 'details': results}
    
    def _extract_statements(self, response: str) -> List[str]:
        return [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') 
                if s.strip() and len(s.strip()) > 15]
    
    def _verify(self, statement: str, context: str) -> bool:
        prompt = f"""Context: {context}
Statement: "{statement}"
Can this be inferred from context? YES or NO"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return 'yes' in result.reasoning.lower()[:10]

# Demo
context = "Product costs $99. Ships in 3-5 days. Free shipping over $150."
response = "This costs $99 with free shipping and arrives in 3-5 days."

ragas = RAGASFaithfulness(judge)
result = ragas.calculate(context, response)

print(f"Faithfulness: {result['faithfulness_score']:.2f}")
for i, d in enumerate(result['details'], 1):
    status = "✓" if d['supported'] else "✗"
    print(f"  {status} {d['statement']}")

# %% [markdown]
"""
## Example 14: Answer Relevancy

Measures if answer addresses the question.
"""

# %%
class AnswerRelevancy:
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, answer: str) -> float:
        prompt = f"""Score relevancy 1-5:

Question: {question}
Answer: {answer}

1=Irrelevant, 5=Perfectly relevant"""
        
        result = self.judge.invoke([HumanMessage(content=prompt)])
        return result.score / 5.0

# Demo
ans_rel = AnswerRelevancy(judge)

q = "What are health benefits of exercise?"
good = "Exercise improves cardiovascular health and mental well-being."
bad = "The weather is nice today."

print(f"Good answer relevancy: {ans_rel.calculate(q, good):.2f}")
print(f"Bad answer relevancy: {ans_rel.calculate(q, bad):.2f}")

# %% [markdown]
"""
## Example 15: Context Precision

Evaluates if relevant contexts rank higher in retrieval.
"""

# %%
class ContextPrecision:
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, question: str, contexts: List[str], ground_truth: str) -> float:
        relevance = []
        for ctx in contexts:
            prompt = f"""Is this relevant?

Question: {question}
Context: {ctx}
Answer: {ground_truth}

YES or NO"""
            result = self.judge.invoke([HumanMessage(content=prompt)])
            relevance.append('yes' in result.reasoning.lower()[:10])
        
        precisions = []
        relevant_count = 0
        for k, is_rel in enumerate(relevance, 1):
            if is_rel:
                relevant_count += 1
                precisions.append(relevant_count / k)
        
        return sum(precisions) / len(precisions) if precisions else 0.0

# Demo
ctx_prec = ContextPrecision(judge)

q = "What is the capital of France?"
contexts = [
    "Paris is the capital of France.",
    "France is in Europe.",
    "The Eiffel Tower is in Paris."
]

print(f"Context Precision: {ctx_prec.calculate(q, contexts, 'Paris'):.2f}")

# %% [markdown]
"""
## Example 16: Context Recall

Measures if all necessary information was retrieved.
"""

# %%
class ContextRecall:
    def __init__(self, llm_judge):
        self.judge = llm_judge
    
    def calculate(self, contexts: List[str], ground_truth: str) -> float:
        facts = [s.strip() for s in ground_truth.split('.') if s.strip()]
        combined = " ".join(contexts)
        
        covered = 0
        for fact in facts:
            prompt = f"""Is this fact in the context?

Fact: {fact}
Context: {combined}

YES or NO"""
            result = self.judge.invoke([HumanMessage(content=prompt)])
            if 'yes' in result.reasoning.lower()[:10]:
                covered += 1
        
        return covered / len(facts) if facts else 1.0

# Demo
ctx_recall = ContextRecall(judge)

contexts = ["Paris is the capital of France."]
ground_truth = "Paris is the capital. Population is 2.2 million."

print(f"Context Recall: {ctx_recall.calculate(contexts, ground_truth):.2f}")
print("⚠️ Low recall = missing information")

# %% [markdown]
"""
**RAG Failure Patterns:**
- Low recall → Improve embeddings
- Low precision → Better ranking
- Low faithfulness → Stronger prompts
"""

# %% [markdown]
"""
---
## Part 4: Production & Best Practices
"""

# %% [markdown]
"""
## Example 17: CI/CD Evaluation Gate

Automated quality gates in deployment pipelines.
"""

# %%
def ci_cd_gate(test_cases: list, min_score: float = 0.75) -> dict:
    scores = [4, 5, 3, 4, 5]  # Mock evaluation scores
    avg = sum(scores) / len(scores) / 5
    
    if avg < min_score:
        raise Exception(f"❌ Quality gate failed: {avg:.2f} < {min_score}")
    
    print(f"✅ Quality gate passed: {avg:.2f} ≥ {min_score}")
    return {'avg_score': avg, 'passed': True}

# Demo
result = ci_cd_gate([])
print(f"Average score: {result['avg_score']:.2f}")

# %% [markdown]
"""
## Example 18: Tiered Evaluation Strategy

Cost optimization through sampling.
"""

# %%
class TieredEvaluation:
    @staticmethod
    def fast_check(response: str) -> bool:
        """100% traffic - Basic validation"""
        return 0 < len(response) < 5000
    
    @staticmethod
    def comprehensive(response: str) -> dict:
        """1% sampling - Full LLM evaluation"""
        return {'score': 4, 'passed': True}

# Demo
import random
response = "Sample response"

if TieredEvaluation.fast_check(response):
    print("✅ Fast check passed (100%)")
    
    if random.random() < 0.01:  # 1% sampling
        result = TieredEvaluation.comprehensive(response)
        print(f"✅ Comprehensive: {result}")

# %% [markdown]
"""
## Best Practices Checklist

### Before Deployment:
- [ ] Comprehensive test coverage
- [ ] Clear metrics aligned with goals
- [ ] Automated CI/CD evaluation
- [ ] Safety checks (toxicity, hallucinations)
- [ ] Security testing (prompt injection)

### After Deployment:
- [ ] Continuous monitoring
- [ ] Real-time safety checks
- [ ] Performance tracking
- [ ] User feedback collection
- [ ] Quarterly human calibration
"""

# %% [markdown]
"""
## Summary

### What We Covered:
✅ **Foundations**: P/R/F1, BLEU, ROUGE, BERTScore, Validation  
✅ **LLM-as-Judge**: Simple, Multi-criteria, G-Eval, Toxicity  
✅ **RAG**: Faithfulness, Relevancy, Precision, Recall  
✅ **Production**: CI/CD, Tiered evaluation, Best practices  

### Key Takeaways:
1. **Multi-dimensional** - No single metric captures quality
2. **Continuous** - Evaluation spans entire lifecycle
3. **Hybrid** - Combine traditional, neural, and LLM methods
4. **Human alignment** - Validate against human judgment

### Next Steps:
1. Start with critical metrics for your use case
2. Implement LLM-as-judge for subjective quality
3. Build evaluation into CI/CD
4. Monitor continuously in production
"""

# %%
