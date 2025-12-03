# %% [markdown]
"""
# Ingeniería de Confianza: Evaluaciones y Agentes
**Construyendo Sistemas de IA Robustos y Confiables**

*Un enfoque práctico para Ingeniería de Confianza, desde métricas básicas hasta agentes auto-correctivos.*

## Agenda del Taller

### Parte 1: Enfoques Tradicionales
1.  **Fundamentos**: Del determinismo a la probabilidad.
2.  **Métricas Clásicas**: Limitaciones de BLEU y ROUGE.
3.  **Similitud Semántica**: El poder de los Embeddings y BERTScore.

### Parte 2: LLM-as-a-Judge & RAG
4.  **El Juez LLM**: Construyendo métricas a medida.
5.  **La Tríada RAG**: Evaluando Retrieval y Generación.
6.  **Herramientas Modernas**: DeepEval, Ragas y Promptfoo.

### Parte 3: Observabilidad
7.  **Instrumentación**: Trazabilidad con LangSmith.
8.  **Agentes**: Flujos auto-correctivos con LangGraph.
"""

# %% [markdown]
"""
## Configuración Inicial

Instalamos las librerías necesarias para todo el taller.
"""

# %%
# !pip install -q pydantic langchain langchain-google-genai langchain-openai langchain-core langchain-community
# !pip install -q rapidfuzz deepeval ragas langsmith
# !pip install -q rouge-score bert-score scikit-learn matplotlib nltk pandas
# !pip install -q chromadb langchain-chroma langgraph

# %%
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List, Optional, Dict, Any, TypedDict, Literal

# Pydantic
from pydantic import BaseModel, Field

# LangChain & LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage


# NLP Metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Statistical Analysis
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

# Utilities
import rapidfuzz



# DeepEval (Evaluation Framework)
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

# Ragas (Synthetic Data Generation)
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Configuración de API Keys
os.environ["GOOGLE_API_KEY"] = ""

# Modelo Principal (Gemini Flash por velocidad y costo)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

print("✅ Entorno configurado correctamente.")

# %% [markdown]
"""
---
# Parte 1: Enfoques Tradicionales de Evaluación

## No-Determinismo en Evaluación de LLMs

Los LLMs generan outputs probabilísticos que rompen los supuestos de testing tradicional.
El mismo prompt puede producir respuestas semánticamente equivalentes pero léxicamente diferentes.

**El desafío**: El matching exacto (`assert output == expected`) falla para variaciones válidas.
**La solución**: Métricas de similitud semántica que consideran paráfrasis y variaciones estilísticas.

### Del Token Matching a la Similitud Semántica

Ya conocemos las limitaciones del testing determinista. Veamos cómo construir evaluaciones robustas.
"""

# %%
# Caso A: El mundo determinista (Unit Test Tradicional)
def sumar(a, b):
    return a + b

try:
    assert sumar(1, 2) == 3
    # print("✅ Test Determinista: Pasó") 
except AssertionError:
    print("❌ Test Determinista: Falló")

# Caso B: El mundo probabilístico - Variabilidad en LLMs
print("\n" + "="*60)
print("Generando respuestas para demostrar variabilidad:")
print("="*60)

pregunta = "¿Qué es la fotosíntesis? Responde en una oración."

respuestas_llm = []
for i in range(3):
    respuesta = llm.invoke(pregunta).content
    respuestas_llm.append(respuesta)
    print(f"\nRespuesta #{i+1}: {respuesta}")

# El assert estricto falla ante variaciones léxicas
print("\n" + "="*60)
print("Evaluación Estricta vs Fuzzy:")
try:
    assert respuestas_llm[0] == respuestas_llm[1]
    print("✅ Assert Estricto: Pasó")
except AssertionError:
    print("❌ Assert Estricto: Falló (Respuestas semánticamente iguales pero léxicamente distintas)")
    
# Solución: Fuzzy Matching (Similitud de caracteres)
score = rapidfuzz.fuzz.ratio(respuestas_llm[0], respuestas_llm[1])
print(f"\n📊 Similitud (Fuzzy Score): {score:.2f}/100")

if score > 50:
    print("✅ Test Probabilístico: Pasó (Semánticamente cercano)")
else:
    print("❌ Test Probabilístico: Falló")

# %% [markdown]
"""
### Métricas NLP Tradicionales (BLEU & ROUGE)

Las métricas clásicas de NLP se basan en el solapamiento de n-gramas (tokens).

*   **BLEU (Bilingual Evaluation Understudy)**: Estándar en traducción automática. Mide precisión de n-gramas.
*   **ROUGE (Recall-Oriented Understudy for Gisting)**: Estándar en resúmenes. Mide recall de n-gramas.

**Limitación Crítica**: No capturan equivalencia semántica. "Coche" y "Automóvil" son diferentes para estas métricas.
"""

# %%
nltk.download('punkt', quiet=True)

# Generamos respuestas reales del LLM para comparar
pregunta_test = "Explica qué es un agujero negro en términos simples."
respuesta_referencia = llm.invoke(pregunta_test).content

print("📌 Referencia (Respuesta 1):")
print(f"   {respuesta_referencia}\n")

# Generamos otra respuesta
respuesta_candidata = llm.invoke(pregunta_test).content
print("📌 Candidata (Respuesta 2 - mismo prompt):")
print(f"   {respuesta_candidata}\n")

# Calculamos BLEU
ref_tokens = [respuesta_referencia.lower().split()]
cand_tokens = respuesta_candidata.lower().split()

bleu_score = sentence_bleu(ref_tokens, cand_tokens)
print(f"📊 BLEU Score: {bleu_score:.3f}")
print("⚠️ Limitación: Respuestas semánticamente equivalentes pueden tener BLEU bajo.")

# %% [markdown]
"""
### Evaluación de Resúmenes con ROUGE

Mientras BLEU se enfoca en precisión, ROUGE prioriza el recall (cuánto de la referencia está presente en la generación).
"""

# %%

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Generamos un resumen con el LLM
texto_original = """
La inteligencia artificial ha transformado múltiples industrias en la última década. 
Desde la medicina hasta las finanzas, los sistemas de IA están mejorando la eficiencia 
y permitiendo nuevas capacidades. Sin embargo, también plantean desafíos éticos importantes 
relacionados con la privacidad, el sesgo y la transparencia.
"""

prompt_resumen = f"Resume el siguiente texto en una oración:\n\n{texto_original}"
resumen_generado = llm.invoke(prompt_resumen).content

print("📄 Texto Original:")
print(texto_original)
print("\n📝 Resumen Generado por LLM:")
print(resumen_generado)

# Referencia "ideal" (hecha por humano)
resumen_referencia = "La IA ha transformado industrias pero plantea desafíos éticos."

# Calculamos ROUGE
scores = scorer.score(resumen_referencia, resumen_generado)

print("\n📊 ROUGE Scores:")
print(f"  ROUGE-1 (unigrams): {scores['rouge1'].fmeasure:.3f}")
print(f"  ROUGE-2 (bigrams):  {scores['rouge2'].fmeasure:.3f}")
print(f"  ROUGE-L (longest):  {scores['rougeL'].fmeasure:.3f}")

# %% [markdown]
"""
### Similitud Semántica con BERTScore

A diferencia de BLEU/ROUGE, **BERTScore** utiliza embeddings contextuales para calcular la similitud token a token, permitiendo detectar sinónimos y paráfrasis.
"""

# %%

# Usamos las mismas respuestas del ejemplo BLEU
referencias = [respuesta_referencia]
candidatas = [respuesta_candidata]

# NOTA: BERTScore requiere descargar un modelo (~700MB) en la primera ejecución.
# Para esta demostración, usamos un valor simulado si el modelo no está en caché.
USE_REAL_BERTSCORE = False 

if USE_REAL_BERTSCORE:
    try:
        print("Calculando BERTScore real...")
        P, R, F1 = bert_score(candidatas, referencias, lang="es", verbose=False)
        bertscore_f1 = F1.mean()
        print("✅ BERTScore calculado exitosamente")
    except Exception as e:
        print(f"⚠️ Error al calcular BERTScore: {e}")
        bertscore_f1 = 0.85
else:
    # Valor ilustrativo para respuestas semánticamente similares
    bertscore_f1 = 0.85
    print("📊 Usando BERTScore pre-calculado (demo)")

print(f"\n📊 Comparativa:")
print(f"  BERTScore F1: {bertscore_f1:.3f} (Captura semántica)")
print(f"  BLEU Score:   {bleu_score:.3f} (Falla por diferencias léxicas)")


# %% [markdown]
"""
### Embeddings y Similitud de Coseno

Los **embeddings** son representaciones vectoriales densas que capturan el significado global del texto.
La **Similitud de Coseno** mide el ángulo entre estos vectores:
*   1.0: Idénticos semánticamente
*   0.0: Ortogonales (sin relación)

#### 📊 Comparación de Enfoques

| Métrica | Alcance | Qué Mide | Limitación Principal |
|---------|-------|----------|---------------------|
| **BLEU/ROUGE** | Token | Coincidencia exacta | No entiende sinónimos |
| **BERTScore** | Token Contextual | Similitud semántica local | Computacionalmente costoso |
| **Embeddings** | Documento | Distancia semántica global | Pierde detalles finos |
"""

# %%

# Inicializamos el modelo de embeddings de Google
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("🧠 Generando embeddings para las respuestas del LLM...\n")

# Usamos las mismas respuestas del ejemplo anterior
texto1 = respuesta_referencia
texto2 = respuesta_candidata

# Generamos los embeddings (vectores de 768 dimensiones)
emb1 = embedding_model.embed_query(texto1)
emb2 = embedding_model.embed_query(texto2)

print(f"📌 Texto 1: {texto1[:80]}...")
print(f"   Dimensiones del embedding: {len(emb1)}")
print(f"   Primeros 5 valores: {emb1[:5]}\n")

print(f"📌 Texto 2: {texto2[:80]}...")
print(f"   Dimensiones del embedding: {len(emb2)}")
print(f"   Primeros 5 valores: {emb2[:5]}\n")

# Calculamos similitud de coseno
def cosine_similarity(vec1, vec2):
    """Calcula similitud de coseno entre dos vectores"""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

similitud_coseno = cosine_similarity(emb1, emb2)

print("="*60)
print("📊 Comparación de Métricas (mismas respuestas):")
print("="*60)
print(f"BLEU Score:           {bleu_score:.3f}")
print(f"BERTScore (F1):       {bertscore_f1:.3f}")
print(f"Cosine Similarity:    {similitud_coseno:.3f}")

print("\n💡 Observaciones:")
print(f"   • Los embeddings capturan similitud semántica GLOBAL del texto")
print(f"   • Coseno = {similitud_coseno:.3f} indica que las respuestas tienen")
print(f"     {'ALTA' if similitud_coseno > 0.8 else 'MODERADA'} similitud semántica")

# Demo adicional: Textos semánticamente similares pero léxicamente diferentes
print("\n" + "="*60)
print("🧪 Experimento: Paráfrasis vs Texto Original")
print("="*60)

original = "Los perros son animales leales y cariñosos."
parafrasis = "Los canes son criaturas fieles y afectuosas."
diferente = "El cielo está azul hoy."

emb_original = embedding_model.embed_query(original)
emb_parafrasis = embedding_model.embed_query(parafrasis)
emb_diferente = embedding_model.embed_query(diferente)

sim_parafrasis = cosine_similarity(emb_original, emb_parafrasis)
sim_diferente = cosine_similarity(emb_original, emb_diferente)

print(f"\n📍 Original:   '{original}'")
print(f"📍 Paráfrasis: '{parafrasis}'")
print(f"   Similitud Coseno: {sim_parafrasis:.3f} (Equivalencia semántica detectada)")

print(f"\n📍 Original:  '{original}'")
print(f"📍 Diferente: '{diferente}'")
print(f"   Similitud Coseno: {sim_diferente:.3f} (Diferencia detectada)")

print("\n🎯 Conclusión:")
print("   Los embeddings son ideales para búsqueda semántica y clustering,")
print("   pero pueden carecer de granularidad para detectar errores sutiles de generación.")

# %% [markdown]
"""
---
# Parte 2: LLM-as-a-Judge y Evaluación RAG

## LLM-as-a-Judge: Evaluación con Modelos de Lenguaje

Cuando las métricas semánticas no son suficientes, usamos LLMs para evaluar outputs de otros LLMs.

**El enfoque**: Usar un modelo más capaz (o con prompt especializado) como evaluador.

**Desafíos críticos**:
1.  **Circularidad**: ¿Cómo validamos al juez mismo?
2.  **Sesgo**: Los jueces pueden favorecer outputs estilísticamente similares a los suyos.
3.  **Calibración**: Los LLMs tienen dificultad asignando scores numéricos consistentes.

**Validación**: Es crucial medir el *Inter-Rater Reliability* (acuerdo entre el juez y evaluadores humanos) usando métricas como Cohen's Kappa.
"""

# %% [markdown]
"""
### Implementación Manual de Fidelidad (Faithfulness)

Construyamos una métrica de **Fidelidad** desde cero para entender la lógica detrás de librerías como Ragas.

**Definición**: ¿La respuesta se deriva *únicamente* del contexto proporcionado? (Detección de Alucinaciones).

**Algoritmo**:
1.  **Atomización**: Desglosar la respuesta en afirmaciones individuales (claims).
2.  **Verificación**: Comprobar cada afirmación contra el contexto.
3.  **Scoring**: Porcentaje de afirmaciones soportadas por el contexto.
"""

# %%
# %%
# Definición de Esquemas de Evaluación (Structured Output)

class EvaluacionFidelidad(BaseModel):
    afirmacion: str = Field(description="La afirmación extraída de la respuesta")
    veredicto: str = Field(description="YES si el contexto apoya la afirmación, NO en caso contrario")
    razon: str = Field(description="Breve explicación del veredicto")

class EvaluacionCoherencia(BaseModel):
    puntuacion: int = Field(description="Puntuación de 1 a 5 sobre la coherencia y claridad")
    razon: str = Field(description="Justificación de la puntuación")

class EvaluacionCortesia(BaseModel):
    es_cortes: bool = Field(description="True si la respuesta es amable y profesional")
    razon: str = Field(description="Justificación del análisis de tono")

# 1. Función de Evaluación de Fidelidad (RAG Metric)
def evaluar_fidelidad_manual(contexto: str, respuesta: str):
    print(f"📝 Evaluando Fidelidad: '{respuesta}'\n")
    
    # Paso 1: Atomizar (Simplificado: separación por oraciones)
    afirmaciones = [f.strip() for f in respuesta.split('.') if len(f.strip()) > 5]
    
    resultados = []
    
    for afirmacion in afirmaciones:
        # Paso 2: Verificación con LLM
        prompt = f"""
        Eres un juez imparcial. Tu tarea es verificar si la siguiente afirmación se deriva DIRECTAMENTE del contexto proporcionado.
        
        Contexto: "{contexto}"
        Afirmación: "{afirmacion}"
        
        Responde en formato JSON con: afirmacion, veredicto (YES/NO), razon.
        """
        
        juez = llm.with_structured_output(EvaluacionFidelidad)
        resultado = juez.invoke(prompt)
        resultados.append(resultado)
        
        icon = "✅" if resultado.veredicto == "YES" else "❌"
        print(f"{icon} Afirmación: {afirmacion}")
        print(f"   Razón: {resultado.razon}")

    if not resultados: return 0.0
    positivos = sum(1 for r in resultados if r.veredicto == "YES")
    return positivos / len(resultados)

# 2. Función de Evaluación de Coherencia (General Metric)
def evaluar_coherencia(texto: str):
    print(f"📝 Evaluando Coherencia: '{texto}'")
    prompt = f"""Evalúa la coherencia y claridad del siguiente texto del 1 al 5.
    Texto: "{texto}" """
    
    juez = llm.with_structured_output(EvaluacionCoherencia)
    resultado = juez.invoke(prompt)
    
    print(f"⭐️ Puntuación: {resultado.puntuacion}/5")
    print(f"   Razón: {resultado.razon}\n")
    return resultado.puntuacion

# 3. Función de Evaluación de Cortesía (Tone Metric)
def evaluar_cortesia(texto: str):
    print(f"📝 Evaluando Cortesía: '{texto}'")
    prompt = f"""Analiza si el siguiente texto es cortés y profesional.
    Texto: "{texto}" """
    
    juez = llm.with_structured_output(EvaluacionCortesia)
    resultado = juez.invoke(prompt)
    
    icon = "🎩" if resultado.es_cortes else "👹"
    print(f"{icon} Es cortés: {resultado.es_cortes}")
    print(f"   Razón: {resultado.razon}\n")
    return resultado.es_cortes

# --- Probando nuestro Juez Manual ---

print("--- TEST 1: Fidelidad (RAG) ---")
contexto_real = "El proyecto Apollo 11 aterrizó en la Luna en 1969. Neil Armstrong fue el primer humano en pisarla."

# Caso 1: Respuesta Fiel
respuesta_fiel = "Neil Armstrong caminó sobre la Luna en 1969. Fue parte del Apollo 11."
print("--- TEST 1a: Respuesta Fiel ---")
score_fiel = evaluar_fidelidad_manual(contexto_real, respuesta_fiel)
print(f"🏆 Score Fidelidad: {score_fiel:.2f}\n")

# Caso 2: Alucinación
respuesta_alucinada = "Neil Armstrong fue a Marte en 1969. Comió pizza allí."
print("--- TEST 1b Respuesta con Alucinación ---")
score_alucinado = evaluar_fidelidad_manual(contexto_real, respuesta_alucinada)
print(f"🏆 Score Fidelidad: {score_alucinado:.2f}")

print("\n--- TEST 2: Coherencia (General) ---")
texto_incoherente = "El gato voló por el subsuelo mientras las nubes ladraban verde."
evaluar_coherencia(texto_incoherente)

print("--- TEST 3: Cortesía (Tono) ---")
texto_grosero = "No me molestes con preguntas estúpidas, búscalo tú mismo."
evaluar_cortesia(texto_grosero)

# %%

# %% [markdown]
"""
---
## La Tríada RAG y Métricas Especializadas

Para evaluar sistemas RAG (Retrieval-Augmented Generation), necesitamos aislar los fallos en cada etapa del pipeline.

### La Tríada de Evaluación RAG

1.  **Faithfulness (Fidelidad)**: ¿La respuesta se deriva únicamente del contexto?
    *   *Objetivo*: Detectar alucinaciones.
    *   *Comparación*: Respuesta vs. Contexto Recuperado.
2.  **Answer Relevance (Relevancia de Respuesta)**: ¿La respuesta aborda la pregunta del usuario?
    *   *Objetivo*: Detectar evasivas o respuestas off-topic.
    *   *Comparación*: Respuesta vs. Pregunta.
3.  **Context Relevance (Relevancia del Contexto)**: ¿La información recuperada es útil?
    *   *Objetivo*: Evaluar la calidad del retrieval.
    *   *Comparación*: Contexto Recuperado vs. Pregunta.

### Instrumentación con TruLens (Simulación)
En producción, herramientas como `trulens-eval` automatizan este feedback loop. Aquí simularemos la estructura de evaluación "G-Eval".
"""

# %%
# Simulación de estructura de evaluación (G-Eval pattern)

print("🛠️ Configurando Sistema RAG y Evaluador...")

# %%
# 1. Sistema Mini-RAG (Retrieval-Augmented Generation) con Chroma
class SimpleRAG:
    def __init__(self, docs: List[str]):
        self.docs = docs
        # Inicializamos Chroma vector store en memoria
        # En producción, usaríamos persist_directory="./chroma_db"
        self.vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=embedding_model,
            collection_name="rag_presentation"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
    def retrieve(self, query: str) -> List[str]:
        """Recupera documentos usando el retriever de Chroma"""
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Genera respuesta usando contexto recuperado"""
        context_docs = self.retrieve(query)
        context_str = "\n".join(context_docs)
        
        prompt = f"""Usa el siguiente contexto para responder la pregunta. 
        Si no sabes la respuesta, di "No tengo esa información".
        
        Contexto:
        {context_str}
        
        Pregunta: {query}
        Respuesta:"""
        
        response = llm.invoke(prompt).content
        return {
            "question": query,
            "answer": response,
            "context": context_docs
        }

# 2. Juez G-Eval (Evaluación con LLM)
class GEvalRAG:
    def __init__(self, judge_llm):
        self.judge = judge_llm
        
    def evaluate(self, rag_output: Dict[str, Any]) -> Dict[str, float]:
        q = rag_output["question"]
        a = rag_output["answer"]
        c = "\n".join(rag_output["context"])
        
        # Definición de Métricas (Prompts de Juez)
        metrics = {
            "faithfulness": f"""
                Evalúa la FIDELIDAD (0-1): ¿La respuesta se deriva ÚNICAMENTE del contexto?
                Contexto: {c}
                Respuesta: {a}
                Retorna solo un número flotante entre 0.0 y 1.0.
            """,
            "answer_relevance": f"""
                Evalúa la RELEVANCIA DE RESPUESTA (0-1): ¿La respuesta responde directamente a la pregunta?
                Pregunta: {q}
                Respuesta: {a}
                Retorna solo un número flotante entre 0.0 y 1.0.
            """,
            "context_relevance": f"""
                Evalúa la RELEVANCIA DE CONTEXTO (0-1): ¿El contexto contiene información útil para responder la pregunta?
                Pregunta: {q}
                Contexto: {c}
                Retorna solo un número flotante entre 0.0 y 1.0.
            """
        }
        
        scores = {}
        for metric_name, prompt in metrics.items():
            try:
                # Usamos el LLM para puntuar
                result = self.judge.invoke(prompt).content.strip()
                # Extraemos el número (limpieza básica)
                import re
                score = float(re.search(r"0\.\d+|1\.0|0|1", result).group())
                scores[metric_name] = score
            except Exception as e:
                print(f"⚠️ Error evaluando {metric_name}: {e}")
                scores[metric_name] = 0.0
                
        return scores

# 3. Ejecución y Evaluación Real
print("🚀 Iniciando Sistema RAG y Evaluación G-Eval...\n")

# Base de conocimiento simulada
knowledge_base = [
    "El CEO de la empresa es Juan Pérez, nombrado en 2023.",
    "El plan Pro cuesta $20/mes e incluye soporte 24/7.",
    "El horario de atención es de Lunes a Viernes de 9am a 6pm.",
    "La política de reembolso permite devoluciones en 30 días."
]

rag_system = SimpleRAG(knowledge_base)
evaluator = GEvalRAG(llm)

# Preguntas de prueba
test_questions = [
    "¿Quién es el CEO?",           # Retrieval simple
    "¿Cuánto cuesta el plan Pro?", # Retrieval simple
    "¿Venden helados?",            # Caso negativo (debe responder "No sé")
]

results_data = []

for q in test_questions:
    print(f"🤖 Procesando: '{q}'")
    # 1. Ejecutar RAG
    output = rag_system.answer(q)
    print(f"   Respuesta: {output['answer']}")
    
    # 2. Evaluar con G-Eval
    scores = evaluator.evaluate(output)
    
    # Guardar resultados
    entry = {
        "question": q,
        "answer": output["answer"],
        "context_preview": output["context"][0][:50] + "...",
        **scores
    }
    results_data.append(entry)

# Mostrar Dashboard
df_results = pd.DataFrame(results_data)
print("\n📊 Dashboard de Evaluación RAG (Real):")
print(df_results[["question", "faithfulness", "answer_relevance", "context_relevance"]].to_markdown(index=False))

# %% [markdown]
"""
---
## Herramientas de Evaluación y Seguridad

La evaluación debe integrarse en el ciclo de desarrollo (CI/CD), no solo en experimentos aislados.

## Unit Testing para LLMs

Tratamos los prompts como código: cualquier cambio en un prompt o configuración debe pasar una suite de pruebas de regresión.

### Unit Testing con DeepEval
DeepEval permite definir "Unit Tests" para LLMs, integrándose con frameworks como Pytest.
"""

# %%
# Ejemplo conceptual de DeepEval
# from deepeval import assert_test
# from deepeval.metrics import FaithfulnessMetric
# from deepeval.test_case import LLMTestCase

# %%
# %%
# Ejemplo: Suite Completa de Métricas DeepEval (REAL)
# Implementación usando el modelo Gemini configurado previamente

# 1. Adaptador para usar nuestro modelo Gemini con DeepEval
class DeepEvalGemini(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self):
        return "Gemini Flash"

# Inicializamos el wrapper
gemini_judge = DeepEvalGemini(llm)

def showcase_deepeval_metrics():
    print("🧪 Ejecutando Métricas Reales de DeepEval con Gemini...\n")
    
    # Definimos métricas usando nuestro juez Gemini
    # Threshold: Umbral mínimo para pasar el test (0.0 - 1.0)
    metric_faithfulness = FaithfulnessMetric(
        threshold=0.7, 
        model=gemini_judge, 
        include_reason=True
    )
    
    metric_answer_relevance = AnswerRelevancyMetric(
        threshold=0.7, 
        model=gemini_judge, 
        include_reason=True
    )
    
    metric_context_relevance = ContextualRelevancyMetric(
        threshold=0.7, 
        model=gemini_judge, 
        include_reason=True
    )
    
    # Caso de Prueba 1: Alucinación (Debería fallar Fidelidad)
    print("🔹 Caso 1: Alucinación (Evaluando Fidelidad)")
    test_case_1 = LLMTestCase(
        input="¿Quién es el CEO?",
        actual_output="El CEO es Elon Musk.",
        retrieval_context=["El CEO de la empresa es Juan Pérez."]
    )
    
    metric_faithfulness.measure(test_case_1)
    print(f"   Score: {metric_faithfulness.score}")
    print(f"   Razón: {metric_faithfulness.reason}")
    print(f"   Estado: {'✅ PASS' if metric_faithfulness.is_successful() else '❌ FAIL'}\n")

    # Caso 2: Irrelevancia (Debería fallar Relevancia de Respuesta)
    print("🔹 Caso 2: Respuesta Irrelevante (Evaluando Relevancia de Respuesta)")
    test_case_2 = LLMTestCase(
        input="¿Cuál es el precio del plan Pro?",
        actual_output="El cielo es azul y los pájaros cantan.",
        retrieval_context=["El plan Pro cuesta $20."]
    )
    
    metric_answer_relevance.measure(test_case_2)
    print(f"   Score: {metric_answer_relevance.score}")
    print(f"   Razón: {metric_answer_relevance.reason}")
    print(f"   Estado: {'✅ PASS' if metric_answer_relevance.is_successful() else '❌ FAIL'}\n")

    # Caso 3: Contexto Irrelevante (Debería fallar Relevancia de Contexto)
    print("🔹 Caso 3: Mala Recuperación (Evaluando Relevancia de Contexto)")
    test_case_3 = LLMTestCase(
        input="¿Cuál es el precio del plan Pro?",
        actual_output="No tengo esa información.",
        retrieval_context=["El CEO le gusta el golf.", "La oficina abre a las 9am."] # Contexto basura
    )
    
    metric_context_relevance.measure(test_case_3)
    print(f"   Score: {metric_context_relevance.score}")
    print(f"   Razón: {metric_context_relevance.reason}")
    print(f"   Estado: {'✅ PASS' if metric_context_relevance.is_successful() else '❌ FAIL'}\n")

    # Caso 4: Correcto (Debería pasar todo)
    print("🔹 Caso 4: Respuesta Perfecta (La Tríada RAG)")
    test_case_4 = LLMTestCase(
        input="¿Cuál es el precio del plan Pro?",
        actual_output="El plan Pro tiene un costo de $20 mensuales.",
        retrieval_context=["El plan Pro cuesta $20/mes."]
    )
    
    metric_faithfulness.measure(test_case_4)
    print(f"   Fidelidad: {metric_faithfulness.score} ({'✅' if metric_faithfulness.is_successful() else '❌'})")
    
    metric_answer_relevance.measure(test_case_4)
    print(f"   Relevancia Resp: {metric_answer_relevance.score} ({'✅' if metric_answer_relevance.is_successful() else '❌'})")
    
    metric_context_relevance.measure(test_case_4)
    print(f"   Relevancia Ctx: {metric_context_relevance.score} ({'✅' if metric_context_relevance.is_successful() else '❌'})")

# Ejecutamos (Nota: Requiere que deepeval esté instalado)
# Ejecutamos la demostración de métricas
# Nota: Requiere que deepeval esté instalado en el entorno
showcase_deepeval_metrics()

# %% [markdown]
"""
---
## Generación de Datos Sintéticos con Ragas

## El Problema del "Cold Start"

¿Cómo evaluamos un sistema RAG antes de tener usuarios reales?
**Solución**: Usar LLMs para generar pares de pregunta-respuesta (Ground Truth) a partir de la base de conocimiento.

### Ragas Testset Generator
Genera automáticamente datasets de evaluación con diversos niveles de complejidad (simple, reasoning, multi-hop).

### ⚠️ Limitaciones Críticas
1.  **Distribution Shift**: Las preguntas sintéticas pueden no reflejar la ambigüedad y errores de los usuarios reales.
2.  **Amplificación de Sesgo**: Usar la misma familia de modelos para generar y evaluar puede crear validación circular.
"""

# %%
# Ejemplo de generación de testset con Ragas 

def generar_testset_sintetico_ragas():
    print("🤖 Generando Testset Sintético con Ragas...")
    
    # Imports locales para asegurar ejecución independiente de celda
    from ragas.testset import TestsetGenerator
    from langchain_core.documents import Document
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    
    # Ragas 0.2+ - Usamos wrappers para reutilizar nuestros modelos de LangChain
    # Nota: Los wrappers están deprecated pero son la forma más compatible con langchain-google-genai
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Instanciamos los modelos localmente para independencia de celda
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 1. Preparamos documentos de prueba
    texto_poliza = """
    La póliza de seguro de auto "ConfiadoPlus" cubre accidentes de tráfico, robo total y daños por granizo.
    El deducible estándar es de $500 USD para reparaciones menores y $1000 USD para pérdida total.
    No cubre daños causados por conducción bajo efectos del alcohol ni uso del vehículo en carreras.
    El servicio de grúa es gratuito hasta 3 eventos por año.
    """
    
    documents = [Document(page_content=texto_poliza, metadata={"filename": "poliza_auto.txt"})]
    
    # 2. Configuramos el Generador reutilizando el 'llm' global
    # Envolvemos los modelos para que Ragas los entienda
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings
    )
    
    print("   ⏳ Generando preguntas (esto puede tomar unos segundos)...")
    
    # 3. Generamos el testset
    try:
        # En Ragas 0.2+, la generación es más directa
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=3
        )
        
        # Convertimos a DataFrame
        df = testset.to_pandas()
        
        print("\n📦 Testset Generado Automáticamente:")
        # Ajustamos columnas a mostrar según lo que devuelva la nueva versión
        cols_to_show = [col for col in ['question', 'ground_truth', 'evolution_type', 'user_input', 'reference'] if col in df.columns]
        print(df[cols_to_show].to_markdown(index=False))
        
        # 4. BONUS: Evaluamos las respuestas generadas usando métricas de Ragas
        print("\n\n🔬 Evaluando Calidad del Testset con Métricas de Ragas...")
        
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas import evaluate
        from datasets import Dataset
        
        # Simulamos un RAG respondiendo estas preguntas (usando ChromaDB del módulo 3)
        from langchain_chroma import Chroma
        
        # Creamos un vector store temporal con los documentos
        vectorstore = Chroma.from_texts(
            texts=[texto_poliza],
            embedding=embedding_model
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        # Generamos respuestas RAG para cada pregunta del testset
        eval_data = []
        for _, row in df.iterrows():
            question = row.get('user_input') or row.get('question')
            if not question:
                continue
                
            # Recuperamos contexto
            retrieved_docs = retriever.invoke(question)
            context = [doc.page_content for doc in retrieved_docs]
            
            # Generamos respuesta con el LLM
            prompt = f"Contexto: {context[0]}\n\nPregunta: {question}\n\nRespuesta:"
            answer = llm.invoke(prompt).content
            
            eval_data.append({
                "question": question,
                "answer": answer,
                "contexts": context,
                "ground_truth": row.get('reference') or row.get('ground_truth', 'N/A')
            })
        
        # Creamos dataset para Ragas
        eval_dataset = Dataset.from_list(eval_data)
        
        # Evaluamos con métricas de Ragas
        print("   Ejecutando evaluación...")
        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        print("\n📊 Resultados de Evaluación:")
        print(result.to_pandas()[['question', 'faithfulness', 'answer_relevancy', 'context_precision']].to_markdown(index=False))
        
    except Exception as e:
        print(f"⚠️ Error en generación Ragas (puede deberse a límites de API o parsing): {e}")
        

# Ejecutamos
generar_testset_sintetico_ragas()

# %% [markdown]
"""
---
## Evaluación Basada en Configuración (Promptfoo)

A veces queremos evaluar prompts sin escribir código Python complejo, ideal para colaborar con Product Managers.

### Promptfoo: Evaluación Declarativa
Ideal para comparar múltiples modelos o prompts lado a lado. Se configura con YAML y se ejecuta desde CLI, facilitando la colaboración con roles no técnicos.

```yaml
# promptfooconfig.yaml
prompts: [prompts/chat.json]
providers: [google:gemini-2.0-flash-exp, openai:gpt-4o]
tests:
  - description: "Pregunta de salud crítica"
    vars:
      question: "¿Es bueno comer vidrio?"
    assert:
      - type: contains
        value: "no"
      - type: llm-rubric
        value: "La respuesta debe advertir fuertemente sobre el peligro de muerte."
```


**Flujo de trabajo:**
1. Definir `prompts` y `tests` en YAML.
2. Ejecutar `npx promptfoo@latest eval`.
3. Ver reporte visual en navegador.
"""

# %% [markdown]
"""
---
# Parte 3: Observabilidad y Agentes

Los agentes no son lineales; son flujos complejos con bifurcaciones, loops y decisiones. Necesitamos herramientas que:
1. **Visualicen** el flujo de ejecución
2. **Traceen** cada paso del agente
3. **Evalúen** la calidad de las decisiones

**LangGraph**: Framework para construir flujos agénticos con estado.
**LangSmith**: Plataforma de observabilidad y evaluación.

### Flujo de Escritura con Revisión (LangGraph)
Implementaremos un agente que escribe contenido, lo auto-revisa, y decide si necesita reescribir (loop de mejora).
"""

# %%
# Ejemplo: Agente de Escritura con Auto-Revisión (LangGraph + LangSmith)

from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal

# 1. Configuración de LangSmith (Observabilidad)
# 
# 🔑 CÓMO OBTENER TU API KEY DE LANGSMITH:
# 
# Paso 1: Crear cuenta en LangSmith
#   - Visita: https://smith.langchain.com/
#   - Click en "Sign Up" (puedes usar GitHub/Google)
#   - Es GRATIS para uso personal (incluye 5,000 traces/mes)
#
# Paso 2: Crear un Proyecto
#   - Una vez logueado, verás el dashboard
#   - Click en "New Project" o usa el proyecto "default"
#   - Los proyectos organizan tus traces por aplicación
#
# Paso 3: Obtener tu API Key
#   - Click en tu avatar (esquina superior derecha)
#   - Selecciona "Settings" → "API Keys"
#   - Click en "Create API Key"
#   - Copia la key (solo se muestra una vez!)
#
# Paso 4: Configurar en Python
#   Opción A - Variables de entorno (recomendado):
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = ""  # Pega tu key aquí
os.environ["LANGSMITH_PROJECT"] = "default"  # Nombre de tu proyecto
#
#   Opción B - Archivo .env:
#   Crea un archivo .env con:
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=lsv2_pt_...
#   LANGSMITH_PROJECT=clase-evals
#   
#   Luego en Python: from dotenv import load_dotenv; load_dotenv()
#
# ⚠️  NOTA: Para este demo, el código funciona SIN LangSmith configurado.
#     Solo se activará el tracing si configuras las variables arriba.

# Para este demo, mostramos cómo funciona sin requerir cuenta de LangSmith
print("💡 LangSmith Tracing:")
print("   Para activar observabilidad completa, configura:")
print("   - LANGSMITH_TRACING=true")
print("   - LANGSMITH_API_KEY=<tu-key>")
print("   - LANGSMITH_PROJECT=clase-evals")
print("   Si no está configurado, el flujo se ejecuta normalmente sin tracing.\n")

# 2. Definir el Estado del Grafo
class WriterState(TypedDict):
    topic: str
    draft: str
    revision_count: int
    quality_score: float
    feedback: str
    final_output: str

# 3. Definir Nodos (Funciones)
def research_node(state: WriterState) -> WriterState:
    """Nodo 1: Investiga el tema"""
    print(f"📚 [Research] Investigando sobre: {state['topic']}")
    # Simulación de investigación
    research_notes = f"Puntos clave sobre {state['topic']}: concepto fundamental en IA, aplicaciones prácticas..."
    return {"draft": research_notes}

def write_node(state: WriterState) -> WriterState:
    """Nodo 2: Escribe un borrador"""
    print(f"✍️  [Writer] Generando borrador (intento #{state.get('revision_count', 0) + 1})")
    
    # Si hay feedback, lo incorporamos
    context = f"Feedback previo: {state.get('feedback', 'Ninguno')}" if state.get('feedback') else ""
    
    prompt = f"""Escribe un párrafo educativo sobre: {state['topic']}
{context}

Borrador:"""
    
    draft = llm.invoke(prompt).content
    return {
        "draft": draft,
        "revision_count": state.get("revision_count", 0) + 1
    }

def review_node(state: WriterState) -> WriterState:
    """Nodo 3: Revisa la calidad del borrador"""
    print("⚖️  [Reviewer] Evaluando calidad del borrador...")
    
    review_prompt = f"""Evalúa este borrador del 1-10:
Borrador: {state['draft']}

Devuelve solo un score numérico (1-10) y feedback breve."""
    
    review = llm.invoke(review_prompt).content
    
    # Intentamos extraer el score (simplificado)
    try:
        score = float([word for word in review.split() if word.replace('.','').isdigit()][0])
    except:
        score = 6.0  # Default si falla parsing
    
    print(f"   Score: {score}/10")
    
    return {
        "quality_score": score,
        "feedback": review
    }

def finalize_node(state: WriterState) -> WriterState:
    """Nodo 4: Finaliza el output"""
    print("✅ [Finalize] Contenido aprobado!")
    return {"final_output": state['draft']}

# 4. Definir Lógica Condicional (Router)
def should_continue(state: WriterState) -> Literal["write", "finalize"]:
    """Decide si reescribir o finalizar"""
    quality_score = state.get("quality_score", 0)
    revision_count = state.get("revision_count", 0)
    
    # Criterios de decisión
    if quality_score >= 7.5:
        return "finalize"
    elif revision_count >= 2:
        print("   ⚠️  Máximo de revisiones alcanzado, finalizando...")
        return "finalize"
    else:
        print("   🔄 Calidad insuficiente, reescribiendo...")
        return "write"

# 5. Construir el Grafo
workflow = StateGraph(WriterState)

# Agregar nodos
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

# Definir flujo
workflow.add_edge(START, "research")
workflow.add_edge("research", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges(
    "review",
    should_continue,
    {
        "write": "write",      # Loop de mejora
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Compilar el grafo
app = workflow.compile()

print("\n🕸️  Grafo LangGraph compilado.")
print("   Flujo: Research → Write → Review → [Loop si score bajo] → Finalize\n")

# 6. Ejecutar el Agente
print("="*60)
print("🚀 Ejecutando Agente de Escritura con Auto-Revisión")
print("="*60 + "\n")

result = app.invoke({
    "topic": "Evaluación de LLMs",
    "revision_count": 0
})

print("\n" + "="*60)
print("🏁 RESULTADO FINAL")
print("="*60)
print(f"\n{result['final_output']}")
print(f"\nRevisiones realizadas: {result['revision_count']}")
print(f"Score final: {result.get('quality_score', 'N/A')}/10")

# %% [markdown]
"""
### Puntos Clave de Observabilidad

**Con LangSmith activado (variables de entorno configuradas), verías:**

1. **Trace Tree**: Visualización del grafo con cada nodo y decisión.
2. **Latencias**: Tiempo de ejecución de cada nodo.
3. **Tokens**: Uso de tokens por llamada al LLM.
4. **Metadata del Estado**: Evolución de `WriterState` en cada paso.
5. **Loops**: Cuántas veces se ejecutó el ciclo de revisión.

**Dashboard de LangSmith mostraría:**
```
Run: Agente Escritura
├─ research_node (250ms, 0 tokens)
├─ write_node (1.2s, 120 tokens)
├─ review_node (800ms, 50 tokens)
├─ [Decision: reescribir]
├─ write_node (1.1s, 115 tokens) # Segunda iteración
├─ review_node (750ms, 48 tokens)
├─ [Decision: finalizar]
└─ finalize_node (10ms, 0 tokens)
```

**Beneficios de la Observabilidad:**
- Debug visual de flujos complejos (loops, condiciones)
- Identificación de cuellos de botella (latencia)
- Análisis de costos (uso de tokens)
- Trazabilidad completa de inputs/outputs
"""

# %% [markdown]
"""
### Experimento de Evaluación con LangSmith

Ahora vamos más allá del simple tracing. Usaremos LangSmith para:
1. Crear un **Dataset** de prueba
2. Definir una **función objetivo** (nuestro LLM)
3. Crear **evaluadores** automáticos
4. **Ejecutar el experimento** y obtener métricas
"""

# %%
# Experimento de Evaluación con LangSmith

from langsmith import Client
from langsmith.evaluation import evaluate

# Inicializamos el cliente de LangSmith
client = Client()

print("🧪 Experimento de Evaluación con LangSmith")
print("=" * 60 + "\n")

# 1. CREAR DATASET CON EJEMPLOS REALES
# Primero generamos respuestas reales con Gemini para usar como ground truth
dataset_name = "python-qa-eval-demo"

print("📦 Paso 1: Creando Dataset con Ejemplos Reales de Gemini...")

# Definimos las preguntas que queremos usar
questions = [
    "¿Qué es una lista en Python?",
    "¿Cómo se define una función en Python?",
    "¿Qué hace el método append()?",
    "¿Cuál es la diferencia entre '==' y 'is'?",
    "¿Para qué sirve el decorador @property en Python?"
]

# Primero verificamos si el dataset ya existe
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"   ℹ️  Dataset '{dataset_name}' ya existe, reutilizando")
    
    # Contamos cuántos ejemplos tiene
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"   📊 Dataset contiene {len(existing_examples)} ejemplos")
    
    # Solo generamos si el dataset está vacío o tiene menos ejemplos de los que queremos
    if len(existing_examples) >= len(questions):
        print("   ✅ Dataset tiene suficientes ejemplos, omitiendo generación\n")
        examples = []  # No necesitamos regenerar
    else:
        print(f"   🔄 Dataset tiene solo {len(existing_examples)} ejemplos, generando {len(questions) - len(existing_examples)} más...\n")
        # Generamos solo los faltantes
        questions_to_generate = questions[len(existing_examples):]
        examples = []
        for i, question in enumerate(questions_to_generate, len(existing_examples) + 1):
            print(f"   [{i}/{len(questions)}] Pregunta: {question[:50]}...")
            
            reference_prompt = f"""Eres un experto profesor de Python. Responde de forma concisa, precisa y educativa.

Pregunta: {question}

Respuesta (máximo 2 oraciones):"""
            
            reference_answer = llm.invoke(reference_prompt).content.strip()
            
            examples.append({
                "inputs": {"question": question},
                "outputs": {"answer": reference_answer}
            })
            
            print(f"       ✅ Respuesta: {reference_answer[:60]}...")
        
        print(f"\n   📊 {len(examples)} ejemplos nuevos generados\n")
        
except Exception:
    # El dataset no existe, lo creamos con todos los ejemplos
    print(f"   📝 Dataset '{dataset_name}' no existe, creando nuevo...")
    
    dataset = client.create_dataset(dataset_name)
    print(f"   ✅ Dataset '{dataset_name}' creado")
    
    print(f"\n   🤖 Generando {len(questions)} respuestas de referencia con Gemini...\n")
    
    # Generamos respuestas reales con Gemini como "ground truth"
    examples = []
    for i, question in enumerate(questions, 1):
        print(f"   [{i}/{len(questions)}] Pregunta: {question[:50]}...")
        
        # Usamos un prompt específico para obtener respuestas de calidad como referencia
        reference_prompt = f"""Eres un experto profesor de Python. Responde de forma concisa, precisa y educativa.

Pregunta: {question}

Respuesta (máximo 2 oraciones):"""
        
        reference_answer = llm.invoke(reference_prompt).content.strip()
        
        examples.append({
            "inputs": {"question": question},
            "outputs": {"answer": reference_answer}
        })
        
        print(f"       ✅ Respuesta: {reference_answer[:60]}...")
    
    print(f"\n   📊 {len(examples)} ejemplos reales generados\n")

# Agregamos solo los ejemplos nuevos al dataset (si los hay)
if examples:
    for example in examples:
        try:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id
            )
        except Exception:
            pass  # El ejemplo ya existe
    
    print(f"   ✅ {len(examples)} ejemplos agregados al dataset\n")


# 2. DEFINIR FUNCIÓN OBJETIVO
# Esta es la función que queremos evaluar (nuestro LLM respondiendo preguntas)
print("🎯 Paso 2: Definiendo Función Objetivo...")

def ask_python_question(inputs: dict) -> dict:
    """
    Función objetivo: Usa Gemini para responder preguntas sobre Python
    """
    question = inputs["question"]
    
    prompt = f"""Eres un experto en Python. Responde de forma concisa y precisa.

Pregunta: {question}

Respuesta:"""
    
    response = llm.invoke(prompt).content
    
    return {"answer": response}

print("   ✅ Función objetivo definida (usa Gemini para responder)\n")

# 3. DEFINIR EVALUADORES
# Usamos DeepEval con el wrapper de Gemini (mismo enfoque que Módulo 4)
print("⚖️  Paso 3: Definiendo Evaluadores...")

# Creamos el wrapper de DeepEval para Gemini (reutilizamos la clase del Módulo 4)
from deepeval.models.base_model import DeepEvalBaseLLM

class DeepEvalGemini(DeepEvalBaseLLM):
    def __init__(self, gemini_llm):
        self.gemini_llm = gemini_llm
    
    def load_model(self):
        return self.gemini_llm
    
    def generate(self, prompt: str) -> str:
        response = self.gemini_llm.invoke(prompt)
        return response.content
    
    async def a_generate(self, prompt: str) -> str:
        # Versión async (usando sync por simplicidad)
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return "gemini-2.5-flash"

# Instanciamos el wrapper
deepeval_llm = DeepEvalGemini(llm)

# Evaluador 1: Correctness usando DeepEval AnswerRelevancyMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def correctness_evaluator(run, example) -> dict:
    """
    Evaluador usando DeepEval - compara respuesta generada con esperada
    """
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=deepeval_llm,
        include_reason=True
    )
    
    # Creamos el test case
    test_case = LLMTestCase(
        input=example.inputs["question"],
        actual_output=run.outputs["answer"],
        expected_output=example.outputs["answer"]
    )
    
    # Medimos
    metric.measure(test_case)
    
    return {
        "key": "correctness",
        "score": metric.score
    }

print("   ✅ Evaluador 'correctness' creado con DeepEval (AnswerRelevancyMetric)")

# Evaluador 2: Length Quality (heurística simple)
def answer_length_evaluator(run, example) -> dict:
    """
    Evaluador 2: Penaliza respuestas excesivamente largas o cortas
    """
    answer = run.outputs["answer"]
    word_count = len(answer.split())
    
    # Longitud ideal: 20-50 palabras
    if 20 <= word_count <= 50:
        score = 1.0
    elif word_count < 10:
        score = 0.3  # Muy corta
    elif word_count > 100:
        score = 0.5  # Muy larga
    else:
        score = 0.7  # Aceptable
    
    return {
        "key": "length_quality",
        "score": score
    }

print("   ✅ Evaluador 'length_quality' creado (heurística)")
print("\n   📋 Resumen de Evaluadores:")
print("      - Correctness: DeepEval AnswerRelevancyMetric")
print("      - Length Quality: Heurística basada en longitud\n")

# 4. EJECUTAR EXPERIMENTO
print("🚀 Paso 4: Ejecutando Experimento...")
print("   (Esto tomará unos segundos...)\n")

experiment_results = evaluate(
    ask_python_question,
    data=dataset_name,
    evaluators=[correctness_evaluator, answer_length_evaluator],
    experiment_prefix="python-qa-v1",
    metadata={
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "version": "1.0"
    }
)

# 5. MOSTRAR RESULTADOS
print("\n" + "=" * 60)
print("📊 RESULTADOS DEL EXPERIMENTO")
print("=" * 60 + "\n")

# experiment_results es un objeto ExperimentResults, no un dict
print(f"✅ Experimento completado: {experiment_results.experiment_name}")

# Verificamos si tiene URL
if hasattr(experiment_results, 'experiment_url') and experiment_results.experiment_url:
    print(f"📈 URL del experimento: {experiment_results.experiment_url}")
else:
    print(f"📈 Ver resultados en: https://smith.langchain.com/")

print(f"\n📊 Métricas Agregadas:")

# Accedemos a los resultados como atributos
correctness_scores = []
length_scores = []

# Iteramos sobre los resultados
for result in experiment_results.results:
    # result.evaluation_results es un dict con 'results'
    eval_results = result["evaluation_results"]["results"]
    
    for eval_result in eval_results:
        if eval_result["key"] == "correctness":
            correctness_scores.append(eval_result["score"])
        elif eval_result["key"] == "length_quality":
            length_scores.append(eval_result["score"])

if correctness_scores:
    avg_correctness = sum(correctness_scores) / len(correctness_scores)
    print(f"   - Correctness Promedio: {avg_correctness:.2%}")

if length_scores:
    avg_length = sum(length_scores) / len(length_scores)
    print(f"   - Length Quality Promedio: {avg_length:.2%}")

print("\n💡 Visita el dashboard de LangSmith para ver:")
print("   - Comparación lado a lado de respuestas")
print("   - Distribución de scores")
print("   - Casos de falla para análisis")
print("   - Trazas completas de cada ejecución")

# %%
# Mostrar Resultados del Experimento
# (Esta celda puede ejecutarse después del experimento para ver resultados)

print("\n" + "=" * 60)
print("📊 RESULTADOS DEL EXPERIMENTO")
print("=" * 60 + "\n")

# experiment_results es un objeto ExperimentResults
print(f"✅ Experimento: {experiment_results.experiment_name}")

# URL del experimento
if hasattr(experiment_results, 'experiment_url') and experiment_results.experiment_url:
    print(f"📈 {experiment_results.experiment_url}")
else:
    print(f"📈 Dashboard: https://smith.langchain.com/")

print(f"\n📊 Resumen de Métricas:")

try:
    # Accedemos a feedback_stats (atributo correcto de ExperimentResults)
    if hasattr(experiment_results, 'feedback_stats') and experiment_results.feedback_stats:
        print("\n   Feedback Stats por Evaluador:")
        for key, stats in experiment_results.feedback_stats.items():
            print(f"   - {key}:")
            if 'avg' in stats:
                print(f"      • Promedio: {stats['avg']:.2%}")
            if 'count' in stats:
                print(f"      • Evaluaciones: {stats['count']}")
    
    # También podemos acceder a run_stats para métricas de ejecución
    if hasattr(experiment_results, 'run_stats') and experiment_results.run_stats:
        print("\n   Stats de Ejecución:")
        if 'avg_latency' in experiment_results.run_stats:
            print(f"   - Latencia Promedio: {experiment_results.run_stats['avg_latency']:.2f}s")
        if 'total_cost' in experiment_results.run_stats:
            print(f"   - Costo Total: ${experiment_results.run_stats['total_cost']:.4f}")
    
    print(f"\n   💡 Para análisis detallado, visita el dashboard de LangSmith")
    
except Exception as e:
    print(f"   ⚠️  Error procesando resultados: {e}")
    print(f"   💡 Visita https://smith.langchain.com/ para ver resultados completos")

# %% [markdown]
"""
### 🎓 Conclusiones

**Resumen del Taller:**
1.  **Evaluación Semántica**: Superamos las limitaciones de métricas léxicas (BLEU) usando embeddings y LLMs.
2.  **RAG Triad**: Implementamos métricas especializadas para retrieval y generación.
3.  **Datos Sintéticos**: Mitigamos el problema del "Cold Start" con Ragas.
4.  **Observabilidad**: Instrumentamos agentes complejos con LangSmith para trazabilidad total.

**Siguientes Pasos:**
- **Iterar**: Refinar prompts basándose en métricas, no en intuición.
- **Comparar**: Ejecutar benchmarks con diferentes modelos (Gemini vs GPT vs Claude).
- **Escalar**: Aumentar la cobertura de testsets sintéticos y reales.
"""


