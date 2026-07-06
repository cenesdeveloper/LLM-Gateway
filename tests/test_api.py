import pytest
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture
def client():
    # Use TestClient as a context manager so the lifespan runs and
    # app.state.router is created/started before the tests hit it.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion_shape(client):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    body = response.json()
    # OpenAI-compatible shape
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llm_requests_total" in response.text
