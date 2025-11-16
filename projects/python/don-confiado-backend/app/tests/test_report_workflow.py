from fastapi import FastAPI
from fastapi.testclient import TestClient
from endpoints.report_webservice import report_webservice_api_router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(report_webservice_api_router)
    return app


def test_report_workflow_safe():
    app = create_app()
    client = TestClient(app)
    payload = {"message": "Genera un resumen de ventas del último mes", "user_id": "user-safe"}
    resp = client.post("/api/report_v1.0", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "Acme Corp" in data["answer"] or "42" in data["answer"]


def test_report_workflow_unsafe():
    app = create_app()
    client = TestClient(app)
    payload = {"message": "DROP TABLE users;--", "user_id": "user-unsafe"}
    resp = client.post("/api/report_v1.0", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "insegura" in data["answer"] or "No puedo proceder" in data["answer"]


