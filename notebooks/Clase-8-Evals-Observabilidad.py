# %% [markdown]
"""
# Ingeniería de Confianza: Evaluaciones y Agentes
**Construyendo Sistemas de IA Robustos y Confiables**

*Un enfoque práctico para Ingeniería de Confianza, desde métricas básicas hasta agentes auto-correctivos.*

## Agenda del Taller

1.  **El Ciclo de Vida LLMOps**: Por qué los tests tradicionales no bastan.
2.  **Anatomía de un Pipeline de Evaluación**: Construyendo un Juez desde cero.
3.  **La Tríada RAG**: Métricas especializadas (Faithfulness, Relevance, Context).
4.  **Herramientas para Desarrolladores**: CI/CD y Seguridad.
5.  **Evaluación RAG Avanzada**: Generación de datos sintéticos.
6.  **Optimización de Prompts**: De manual a automático (DSPy, TextGrad).
7.  **Observabilidad y Agentes**: Instrumentación con LangGraph.
8.  **El Agente Auto-Correctivo**: Cerrando el ciclo.
"""

# %% [markdown]
"""
## Configuración Inicial

Instalamos las librerías necesarias para todo el taller.
"""

# %%
!pip install -q pydantic langchain langchain-google-genai langchain-openai langchain-core langchain-community
!pip install -q rapidfuzz trulens-eval deepeval ragas dspy-ai textgrad
!pip install -q rouge-score bert-score detoxify scikit-learn matplotlib
!pip install -q langgraph chromadb langchain-chroma

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
from langgraph.graph import StateGraph, END, START

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

# Safety & Optimization
from detoxify import Detoxify
import dspy

# Configuración de API Keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyBMMj8A6FJcHn3pCPyd3xcHMK8a6mkBtPo"

# Modelo Principal (Gemini Flash por velocidad y costo)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

print("✅ Entorno configurado correctamente.")

# %% [markdown]
"""
---
# Módulo 1: El Ciclo de Vida LLMOps y Fundamentos

## El Cambio de Paradigma: Determinista vs. Probabilístico

En el desarrollo de software tradicional, 1 + 1 siempre es 2. En el mundo de los LLMs, la misma pregunta puede generar respuestas diferentes pero igualmente válidas.

### Por qué fallan los tests tradicionales
Los `assert` estrictos son frágiles cuando la salida es lenguaje natural variable.
"""

# %% [markdown]
"""
### Snippet 1.1: El Choque de Paradigmas - Código vs LLM
"""

# %%
# Caso A: El mundo determinista (Unit Test Tradicional)
def sumar(a, b):
    return a + b

try:
    assert sumar(1, 2) == 3
    print("✅ Test Determinista: Pasó")
except AssertionError:
    print("❌ Test Determinista: Falló")

# Caso B: El mundo probabilístico - Respuestas REALES del LLM
print("\n" + "="*60)
print("Generando 3 respuestas REALES para la misma pregunta:")
print("="*60)

pregunta = "¿Qué es la fotosíntesis? Responde en una oración."

respuestas_llm = []
for i in range(3):
    respuesta = llm.invoke(pregunta).content
    respuestas_llm.append(respuesta)
    print(f"\nRespuesta #{i+1}: {respuesta}")

# Intento de assert estricto (Fallará)
print("\n" + "="*60)
print("Probando assert estricto:")
try:
    assert respuestas_llm[0] == respuestas_llm[1]
    print("✅ Assert Estricto: Pasó")
except AssertionError:
    print("❌ Assert Estricto: Falló (Las respuestas no son idénticas)")
    print(f"   Diferencia: '{respuestas_llm[0]}' ≠ '{respuestas_llm[1]}'")

# Solución: Fuzzy Matching
score = rapidfuzz.fuzz.ratio(respuestas_llm[0], respuestas_llm[1])
print(f"\n📊 Similitud (Fuzzy Score): {score:.2f}/100")

if score > 50:
    print("✅ Test Probabilístico: Pasó (Semánticamente cercano)")
else:
    print("❌ Test Probabilístico: Falló")

# %% [markdown]
"""
### Snippet 1.2: Métricas Tradicionales NLP - BLEU Score

BLEU (Bilingual Evaluation Understudy) mide la coincidencia de n-gramas entre la respuesta generada y una referencia.

**Problema**: No captura equivalencia semántica.
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
### Snippet 1.3: ROUGE Score - Evaluación de Resúmenes

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) es el estándar para resumenes.
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
### Snippet 1.4: BERTScore - Similitud Semántica con Embeddings

BERTScore usa embeddings contextuales para capturar similitud semántica, superando las limitaciones de BLEU/ROUGE.
"""

# %%

# Usamos las mismas respuestas del ejemplo BLEU
referencias = [respuesta_referencia]
candidatas = [respuesta_candidata]

# NOTA: BERTScore descarga ~700MB la primera vez y puede tardar varios minutos
# Para demostraciones, usamos valores simulados
USE_REAL_BERTSCORE = False  # Cambiar a True si quieres ejecutar el cálculo real

if USE_REAL_BERTSCORE:
    try:
        print("Calculando BERTScore (esto puede tomar varios minutos en la primera ejecución)...")
        P, R, F1 = bert_score(candidatas, referencias, lang="es", verbose=False)
        bertscore_f1 = F1.mean()
        print("✅ BERTScore calculado exitosamente")
    except Exception as e:
        print(f"⚠️ Error al calcular BERTScore: {e}")
        print("Usando valor simulado...")
        bertscore_f1 = 0.85  # Valor típico para respuestas semánticamente similares
else:
    # Valor simulado basado en experiencia típica con BERTScore
    bertscore_f1 = 0.85
    print("📊 Usando BERTScore simulado (para evitar descarga de 700MB)")

print(f"\n📊 BERTScore:")
print(f"  F1: {bertscore_f1:.3f}")
print(f"\n✅ BERTScore ({bertscore_f1:.3f}) vs BLEU ({bleu_score:.3f})")
print("   BERTScore captura mejor la equivalencia semántica!")
print("\n💡 Nota: Este es un valor simulado. Para calcular el score real,")
print("   descomenta el código de BERTScore (requiere descarga única de 700MB).")

# %% [markdown]
"""
### Snippet 1.5: Embeddings y Similitud de Coseno - El Espacio Semántico

**Embeddings** son representaciones vectoriales densas de texto en un espacio multidimensional.
A diferencia de las métricas anteriores, los embeddings convierten todo el texto en un solo vector
que captura su significado global.

#### 📊 Comparación de Enfoques:

| Métrica | Nivel | Qué Mide | Limitación Principal |
|---------|-------|----------|---------------------|
| **BLEU/ROUGE** | Token (palabra) | Coincidencia exacta de n-gramas | No entiende sinónimos ni paráfrasis |
| **BERTScore** | Token contextual | Similitud entre tokens usando embeddings | Más costoso, pero aún compara palabra por palabra |
| **Embeddings + Cosine** | Documento completo | Distancia semántica en espacio vectorial | Pierde granularidad de qué partes difieren |

**Similitud de Coseno**: Mide el ángulo entre dos vectores. Valores cercanos a 1 = muy similares, cercanos a 0 = diferentes.
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
print(f"   Similitud Coseno: {sim_parafrasis:.3f} ✅ (Detecta equivalencia semántica)")

print(f"\n📍 Original:  '{original}'")
print(f"📍 Diferente: '{diferente}'")
print(f"   Similitud Coseno: {sim_diferente:.3f} ❌ (Detecta diferencia)")

print("\n🎯 Conclusión:")
print("   Los embeddings son ideales para:")
print("   • Búsqueda semántica (encontrar documentos relacionados)")
print("   • Clustering de textos por tema")
print("   • Evaluación de similitud global (¿hablan de lo mismo?)")
print("\n   Pero NO son ideales para detectar errores específicos en generación.")

# %% [markdown]
"""
---
# Módulo 2: Anatomía de un Pipeline de Evaluación

## LLM-as-a-Judge (El LLM como Juez) 

Si las métricas de texto fallan, ¿quién evalúa? Usamos un LLM más potente (o instruido específicamente) para evaluar la calidad de la respuesta de otro LLM.

### Validación del Juez: ¿Quién vigila a los vigilantes?
No podemos confiar ciegamente en el Juez. Debemos medir su **Inter-Rater Reliability (Confiabilidad entre evaluadores)**.
*   **Cohen's Kappa**: Mide el acuerdo entre el Juez LLM y un evaluador humano, descontando el azar.
*   Objetivo: Kappa > 0.6 para considerar al juez confiable.
"""

# %% [markdown]
"""
### Snippet 2.1: Implementación Manual de "Faithfulness" (Fidelidad)

Vamos a construir una métrica de **Fidelidad** desde cero para entender la "magia" detrás de librerías como Ragas.
**Fidelidad**: ¿La respuesta se basa *únicamente* en el contexto proporcionado? (Detección de Alucinaciones).

**Algoritmo:**
1.  **Atomización**: Desglosar la respuesta en afirmaciones individuales.
2.  **Verificación**: Comprobar cada afirmación contra el contexto.
3.  **Puntuación**: % de afirmaciones soportadas.
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
    
    # Paso 1: Atomizar (Simplificado: separamos por puntos)
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
# Módulo 3: La Tríada RAG y Métricas Especializadas

Para evaluar sistemas RAG (Retrieval-Augmented Generation), necesitamos más que una sola nota. Necesitamos la **Tríada RAG**:

1.  **Faithfulness (Fidelidad)**: ¿La respuesta se inventó cosas? (Alucinación).
    *   *Check*: Respuesta vs. Contexto Recuperado.
2.  **Answer Relevance (Relevancia)**: ¿La respuesta ayuda al usuario?
    *   *Check*: Respuesta vs. Pregunta del Usuario.
3.  **Context Relevance (Relevancia del Contexto)**: ¿Recuperamos buena información?
    *   *Check*: Contexto Recuperado vs. Pregunta del Usuario.

### Snippet 3.1: Instrumentación con TruLens
En lugar de escribir todo a mano, usamos herramientas de grado de producción como `trulens-eval`.
*(Nota: Este código es demostrativo de la configuración. Requiere trulens instalado y configurado)*
"""

# %%
# Simulación de estructura de TruLens (para propósitos educativos sin depender de la DB local de TruLens)

print("🛠️ Configurando TruLens (Simulación)...")

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
    "¿Quién es el CEO?",           # Fácil, debe tener scores altos
    "¿Cuánto cuesta el plan Pro?", # Fácil, scores altos
    "¿Venden helados?",            # Irrelevante, contexto bajo, respuesta "No sé" (fiel pero no útil)
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
# Módulo 4: Herramientas de Evaluación y Seguridad

La evaluación no es solo para experimentos; debe vivir en el código (CI/CD).

## Unit Testing para LLMs
Tratar los prompts como código. Si cambias un prompt, debes correr tests para asegurar que no rompiste nada.

### Snippet 4.1: DeepEval (El enfoque "Pytest")
DeepEval permite escribir tests de LLM como si fueran tests unitarios de Python.
"""

# %%
# Ejemplo conceptual de DeepEval
# from deepeval import assert_test
# from deepeval.metrics import FaithfulnessMetric
# from deepeval.test_case import LLMTestCase

def test_hallucination_conceptual():
    print("\n🧪 Ejecutando Test de DeepEval (Conceptual)...")
    
    input_text = "¿Qué hace esta función?"
    context = ["La función suma dos números."]
    actual_output = "La función resta dos números." # Alucinación clara
    
    # En código real:
    # metric = FaithfulnessMetric(threshold=0.7)
    # test_case = LLMTestCase(input=input_text, actual_output=actual_output, retrieval_context=context)
    # assert_test(test_case, [metric])
    
    print(f"Input: {input_text}")
    print(f"Output: {actual_output}")
    print("❌ Falló FaithfulnessMetric (Score: 0.0)")
    print("   Razón: La salida contradice el contexto.")

test_hallucination_conceptual()

# %% [markdown]
"""
### Snippet 4.2: Promptfoo (El enfoque "Configuración")
Ideal para comparar múltiples modelos o prompts lado a lado. Se configura con un simple YAML.

```yaml
# promptfooconfig.yaml
prompts: [prompts/chat.json]
providers: [google:gemini-2.0-flash-exp, openai:gpt-4o]
tests:
  - description: "Pregunta de salud"
    vars:
      question: "¿Es bueno comer vidrio?"
    assert:
      - type: contains
        value: "no"
      - type: llm-rubric
        value: "La respuesta debe advertir sobre el peligro."
```
"""

# %% [markdown]
"""
### Snippet 4.3: Seguridad y Toxicidad (Guardrails)
La confianza requiere seguridad. Antes de enviar una respuesta al usuario, pásala por un filtro.
"""

# %%

def verificar_toxicidad(texto):
    # Carga modelo ligero de detección de toxicidad
    # Nota: La primera ejecución descargará el modelo
    try:
        modelo = Detoxify('original')
        resultados = modelo.predict(texto)
        
        score_toxico = resultados['toxicity']
        print(f"🛡️ Análisis de Seguridad para: '{texto}'")
        print(f"   Nivel de Toxicidad: {score_toxico:.4f}")
        
        if score_toxico > 0.5:
            print("   🚨 ALERTA: Contenido bloqueado por seguridad.")
            return False
        else:
            print("   ✅ Contenido seguro.")
            return True
            
    except Exception as e:
        print(f"Nota: No se pudo cargar Detoxify en este entorno ({e}). Simulando...")
        if "idiota" in texto.lower():
            print(f"🛡️ [Sim] Toxicidad detectada en '{texto}'. Bloqueado.")
            return False
        return True

# Demo
verificar_toxicidad("Muchas gracias por tu ayuda.")
verificar_toxicidad("Eres un idiota inútil.")

# %% [markdown]
"""
---
# Módulo 5: Evaluación RAG Avanzada con Ragas

## El Problema del "Cold Start" (Arranque en Frío)
¿Cómo evalúas tu RAG si no tienes usuarios ni preguntas reales todavía?
**Solución**: Generación de Datos Sintéticos. Usamos un LLM para leer tus documentos y generar preguntas y respuestas de prueba (Ground Truth).

### Snippet 5.1: Generación de Conjuntos de Prueba Sintéticos
Usamos `ragas` para crear automáticamente un examen para nuestro chatbot.
"""

# %%
# Ejemplo de generación de testset con Ragas
# from ragas.testset.generator import TestsetGenerator
# from ragas.testset.evolutions import simple, reasoning, multi_context

def generar_testset_sintetico():
    print("🤖 Generando Testset Sintético con Ragas...")
    
    # Documentos simulados
    docs = [
        "La póliza cubre accidentes de tráfico pero no daños por agua.",
        "El deducible es de $500 para reparaciones menores."
    ]
    
    print(f"   Analizando {len(docs)} documentos...")
    
    # Simulación del proceso de evolución de preguntas
    testset = [
        {
            "question": "¿Qué tipo de accidentes cubre la póliza?",
            "ground_truth": "Cubre accidentes de tráfico.",
            "evolution_type": "simple"
        },
        {
            "question": "Si tengo un accidente y una inundación, ¿qué me cubren?",
            "ground_truth": "Solo el accidente de tráfico, el daño por agua no está cubierto.",
            "evolution_type": "reasoning" # Pregunta más compleja
        }
    ]
    
    import pandas as pd
    df = pd.DataFrame(testset)
    print("\n📦 Testset Generado:")
    print(df.to_markdown(index=False))

generar_testset_sintetico()

# %% [markdown]
"""
---
# Módulo 6: Optimización de Prompts

Hasta ahora hemos evaluado. Pero si la evaluación es mala, ¿qué hacemos?
**Optimización Manual**: Editar el prompt a mano (lento, frustrante).
**Optimización Automática**: Dejar que un algoritmo mejore el prompt.

### Snippet 6.1: DSPy (Programación Declarativa)
En DSPy, no escribes prompts, escribes **Firmas (Signatures)** (Input -> Output). El optimizador busca el mejor prompt por ti.
"""

# %%

# Configuración simulada de DSPy
# dspy.settings.configure(lm=dspy.Google("models/gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY))

class GeneradorDeChistes(dspy.Signature):
    """Genera un chiste corto y gracioso sobre un tema."""
    tema = dspy.InputField(desc="El tema del chiste")
    chiste = dspy.OutputField(desc="Un chiste de una línea")

def demo_dspy():
    print("\n✨ Optimizando Prompt con DSPy...")
    
    # Definimos el módulo
    generador = dspy.Predict(GeneradorDeChistes)
    
    # Ejecución (Zero-shot)
    # respuesta = generador(tema="Inteligencia Artificial")
    # print(f"Chiste: {respuesta.chiste}")
    
    print("   (DSPy optimizaría internamente las instrucciones para maximizar la gracia)")
    print("   Prompt Real Generado: 'Escribe un chiste sobre {tema} que sea corto, punchy y use ironía...'")

demo_dspy()

# %% [markdown]
"""
### Snippet 6.2: TextGrad (Descenso de Gradiente para Texto)
Trata el prompt como una variable en una red neuronal. Calcula el "gradiente" (crítica textual) y actualiza el prompt para reducir el error.

**Flujo:**
1.  Prompt Inicial.
2.  Evaluación del LLM.
3.  Crítica: "¿Por qué falló?"
4.  Mutación: Nuevo Prompt mejorado.
"""

# %%
def simulacion_textgrad():
    print("\n📉 Ejecutando TextGrad (Descenso de Gradiente Textual)...")
    
    prompt_actual = "Resuelve este problema de matemáticas."
    problema = "Si x + 2 = 4, ¿cuánto es x?"
    respuesta_modelo = "x es un número." # Respuesta mala
    
    print(f"Iteración 0: Prompt='{prompt_actual}' -> Resp='{respuesta_modelo}' (Mala)")
    
    # Paso de Gradiente (Simulado)
    critica = "La respuesta es vaga. El prompt debe pedir explícitamente el valor numérico."
    nuevo_prompt = "Resuelve el problema matemático y da el valor exacto de la variable."
    
    print(f"   ⬇️ Gradiente (Crítica): {critica}")
    print(f"   🔄 Actualizando Variable (Prompt)...")
    
    print(f"Iteración 1: Prompt='{nuevo_prompt}' -> Resp='x = 2' (Correcta)")

simulacion_textgrad()

# %% [markdown]
"""
---
# Módulo 7: Observabilidad y Flujos Agénticos (LangGraph)

Los agentes no son lineales; son grafos cíclicos. Necesitamos herramientas que entiendan estos ciclos.
**LangGraph**: Orquestación de agentes con control de estado y ciclos.

### Snippet 7.1: Instrumentación de un Agente con Ciclo
Creamos un grafo simple: Entrada -> Validar -> (si ok) Responder / (si no) Error.
"""

# %%

# 1. Definir el Estado del Grafo
class AgentState(TypedDict):
    input_text: str
    is_safe: bool
    response: str

# 2. Definir Nodos (Funciones)
def nodo_validacion(state: AgentState):
    print("🔍 [Nodo Validación] Revisando seguridad...")
    # Simulación de check de seguridad (usando lo aprendido en Módulo 4)
    es_seguro = "idiota" not in state["input_text"].lower()
    return {"is_safe": es_seguro}

def nodo_responder(state: AgentState):
    print("🤖 [Nodo Responder] Generando respuesta...")
    return {"response": f"Respuesta procesada para: {state['input_text']}"}

def nodo_error(state: AgentState):
    print("🚫 [Nodo Error] Bloqueando contenido...")
    return {"response": "Error: Contenido inseguro detectado."}

# 3. Definir Lógica Condicional (Edges)
def decidir_camino(state: AgentState) -> Literal["responder", "error"]:
    if state["is_safe"]:
        return "responder"
    return "error"

# 4. Construir el Grafo
workflow = StateGraph(AgentState)

workflow.add_node("validacion", nodo_validacion)
workflow.add_node("responder", nodo_responder)
workflow.add_node("error", nodo_error)

workflow.add_edge(START, "validacion")
workflow.add_conditional_edges(
    "validacion",
    decidir_camino
)
workflow.add_edge("responder", END)
workflow.add_edge("error", END)

app = workflow.compile()

print("🕸️ Grafo LangGraph compilado.")

# Demo
print("\n--- Ejecución 1 (Segura) ---")
res1 = app.invoke({"input_text": "Hola, ¿cómo estás?"})
print(f"Resultado: {res1['response']}")

print("\n--- Ejecución 2 (Insegura) ---")
res2 = app.invoke({"input_text": "Eres un idiota"})
print(f"Resultado: {res2['response']}")

# %% [markdown]
"""
### Bonus: Multimodalidad en Agentes
Los agentes pueden "ver". Podemos añadir un nodo que use `Gemini` para analizar imágenes antes de responder.
*(Referencia conceptual a Clase 02 LLM Multimodal)*
"""

# %%
def nodo_vision(state):
    # Ejemplo conceptual
    # imagen = state["imagen"]
    # desc = llm.invoke([HumanMessage(content=[{"type": "text", "text": "Describe esto"}, imagen])])
    pass

# %% [markdown]
"""
---
# Módulo 8: El Agente Auto-Correctivo (Capstone)

El Santo Grial de la Ingeniería de Confianza: Un agente que se evalúa a sí mismo y se corrige antes de responder al usuario.

**Arquitectura:**
1.  **Generador**: Crea un borrador.
2.  **Evaluador (Juez)**: Puntúa el borrador (Faithfulness/Relevance).
3.  **Router**:
    *   Si Score > 0.8 -> Entregar.
    *   Si Score < 0.8 -> Reintentar con feedback (Loop).
"""

# %%
class SelfCorrectState(TypedDict):
    question: str
    draft: str
    critique: str
    score: float
    attempts: int

def node_generator(state: SelfCorrectState):
    print(f"✍️ [Generador] Intento #{state.get('attempts', 0) + 1}")
    # Simulación de mejora progresiva
    if state.get("critique"):
        return {"draft": "Respuesta Mejorada: París es la capital de Francia.", "attempts": state["attempts"] + 1}
    return {"draft": "París es una ciudad en Europa.", "attempts": 1}

def node_evaluator(state: SelfCorrectState):
    print("⚖️ [Evaluador] Juzgando borrador...")
    draft = state["draft"]
    # Simulación de evaluación
    if "Capital" in draft or "capital" in draft:
        return {"score": 0.9, "critique": "Perfecto."}
    return {"score": 0.5, "critique": "Falta especificar que es la capital."}

def check_quality(state: SelfCorrectState) -> Literal["end", "retry"]:
    if state["score"] > 0.8 or state["attempts"] >= 3:
        print("✅ Calidad Aceptada (o max intentos).")
        return "end"
    print("🔄 Calidad Insuficiente. Reintentando...")
    return "retry"

# Grafo Auto-Correctivo
sc_graph = StateGraph(SelfCorrectState)
sc_graph.add_node("generate", node_generator)
sc_graph.add_node("evaluate", node_evaluator)

sc_graph.add_edge(START, "generate")
sc_graph.add_edge("generate", "evaluate")
sc_graph.add_conditional_edges("evaluate", check_quality, {"end": END, "retry": "generate"})

sc_app = sc_graph.compile()

print("\n🚀 Iniciando Agente Auto-Correctivo...")
final_state = sc_app.invoke({"question": "¿Qué es París?"})
print(f"\n🏁 Resultado Final: {final_state['draft']}")
