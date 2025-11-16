# Standard library imports
from typing import TypedDict, List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi_utils.cbv import cbv

# DTOs
from .dto.message_dto import ChatRequestDTO, ChatResponseDTO

# LLM setup (kept for future non-mock nodes)
from langchain.chat_models import init_chat_model

# LangGraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# Mock Tools
from ai.tools.mock_data_tools import mock_supabase_query_tool, mock_neo4j_query_tool

# Logs
from logs.beauty_log import beauty_var_log


class ReportState(TypedDict, total=False):
    user_id: str
    query: str
    messages: List[Any]
    security_flag: bool
    security_notes: str
    retrieved_data: Dict[str, Any]
    report_draft: str
    review_notes: List[str]
    review_severity: str
    final_report: str


report_webservice_api_router = APIRouter()


@cbv(report_webservice_api_router)
class ReportWebService:
    """
    Report workflow service using LangGraph with mocked nodes and tools.
    Entry point is the security check; then an orchestrator routes through:
      collect -> draft -> review -> finalize
    """

    _inMemorySaver = InMemorySaver()

    def __init__(self):
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # Keep LLM allocated for future real nodes; mocks do not use it.
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=self.GOOGLE_API_KEY)

        # Build graph once
        self._compiled_graph = self._build_graph()

    # =========================
    # Graph and Nodes (Mocked)
    # =========================
    def _build_graph(self):
        graph = StateGraph(ReportState)

        # Nodes
        graph.add_node("security_check", self.node_security_check_mock)
        graph.add_node("orchestrator", self.node_orchestrator)
        graph.add_node("collect", self.node_collect_data_mock)
        graph.add_node("draft", self.node_draft_report_mock)
        graph.add_node("review", self.node_adversarial_review_mock)
        graph.add_node("finalize", self.node_finalize)

        # Entry point is security
        graph.set_entry_point("security_check")

        # Routing after security
        graph.add_conditional_edges("security_check", self.route_after_security, {
            "orchestrator": "orchestrator",
            "finalize": "finalize",
        })

        # Orchestrator staged routing
        graph.add_conditional_edges("orchestrator", self.route_from_orchestrator, {
            "collect": "collect",
            "draft": "draft",
            "review": "review",
            "finalize": "finalize",
        })

        # Work nodes return to orchestrator
        graph.add_edge("collect", "orchestrator")
        graph.add_edge("draft", "orchestrator")
        graph.add_edge("review", "orchestrator")

        compiled = graph.compile(checkpointer=self._inMemorySaver)
        return compiled

    # ---------
    # Routers
    # ---------
    def route_after_security(self, state: ReportState) -> str:
        if state.get("security_flag"):
            return "finalize"
        return "orchestrator"

    def route_from_orchestrator(self, state: ReportState) -> str:
        if not state.get("retrieved_data"):
            return "collect"
        if not state.get("report_draft"):
            return "draft"
        if not state.get("review_notes"):
            return "review"
        return "finalize"

    # -----------
    # Node impls
    # -----------
    def node_security_check_mock(self, state: ReportState) -> ReportState:
        query = (state.get("query") or "").lower()
        red_flags = ["drop table", ";--", "jailbreak", "ignore instructions"]
        flagged = any(flag in query for flag in red_flags)
        notes = "Input appears safe." if not flagged else "Potential prompt injection/jailbreak indicators detected."
        return {
            "security_flag": flagged,
            "security_notes": notes,
        }

    def node_orchestrator(self, state: ReportState) -> ReportState:
        # Orchestrator currently is a no-op; routing handled by conditional edges.
        return {}

    def node_collect_data_mock(self, state: ReportState) -> ReportState:
        # Mocked tools simulate successful data retrieval
        supabase_data = mock_supabase_query_tool.invoke({"params": {"period": "last_30_days"}})
        neo4j_data = mock_neo4j_query_tool.invoke({"query": "MATCH (c:Customer)-[r:PURCHASED]->(p:Product) RETURN c.name AS topCustomer, COUNT(r) AS purchases ORDER BY purchases DESC LIMIT 2", "params": {"limit": 2}})
        return {
            "retrieved_data": {
                "supabase": supabase_data,
                "neo4j": neo4j_data,
                "sources": ["supabase", "neo4j"],
            }
        }

    def node_draft_report_mock(self, state: ReportState) -> ReportState:
        query = state.get("query") or ""
        data = state.get("retrieved_data") or {}
        orders = str(((data.get("supabase") or {}).get("orders")) or "N/A")
        revenue = str(((data.get("supabase") or {}).get("revenue")) or "N/A")
        top_customers = ", ".join([row.get("topCustomer", "Unknown") for row in (data.get("neo4j") or [])])
        draft = (
            f"Resumen solicitado: \"{query}\". "
            f"En el último período, se registraron {orders} órdenes con ingresos de {revenue}. "
            f"Clientes destacados: {top_customers or 'Sin datos'}. "
            f"Se observan productos con buen desempeño y oportunidades de seguimiento comercial."
        )
        return {"report_draft": draft}

    def node_adversarial_review_mock(self, state: ReportState) -> ReportState:
        critiques: List[str] = [
            "Verificar si existen anomalías en picos de ventas.",
            "Añadir contexto de estacionalidad para comparar el período.",
            "Validar si clientes destacados mantienen recurrencia."
        ]
        return {
            "review_notes": critiques,
            "review_severity": "low",
        }

    def node_finalize(self, state: ReportState) -> ReportState:
        if state.get("security_flag"):
            return {
                "final_report": "La solicitud parece insegura. No puedo proceder. Reformula tu pregunta de forma segura."
            }
        draft = state.get("report_draft") or ""
        final_text = draft
        # Acknowledge the review step without exposing internals
        if state.get("review_notes"):
            final_text += " Recomendación: considerar estacionalidad y recurrencia de clientes para decisiones."
        return {"final_report": final_text}

    # =========================
    # Endpoint
    # =========================
    @report_webservice_api_router.post("/api/report_v1.0")
    async def generate_report(self, request: ChatRequestDTO):
        """
        Execute the mocked report workflow and return the final report text.
        """
        beauty_var_log("INCOMING REPORT REQUEST", request)

        initial_state: ReportState = {
            "user_id": request.user_id,
            "query": request.message,
            "messages": [],
        }
        config = {"configurable": {"thread_id": request.user_id}}
        final_state: ReportState = self._compiled_graph.invoke(initial_state, config=config)
        beauty_var_log("FINAL REPORT STATE", final_state)

        response_dto = ChatResponseDTO(answer=final_state.get("final_report", "No report available."))
        return response_dto


