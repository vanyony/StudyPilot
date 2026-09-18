from fastapi.testclient import TestClient

from studypilot.api import create_app


def test_planner_vertical_slice(tmp_path) -> None:
    app = create_app(tmp_path / "api.db")
    with TestClient(app) as client:
        _exercise_vertical_slice(client)


def _exercise_vertical_slice(client: TestClient) -> None:

    assert client.get("/health").json() == {"status": "ok"}
    assert client.put(
        "/courses/probability", json={"id": "probability", "name": "概率论"}
    ).status_code == 200
    assert client.put(
        "/courses/probability/topics",
        json=[
            {
                "id": "conditional",
                "course_id": "probability",
                "name": "条件概率",
                "exam_points": 5,
                "learning_minutes": 15,
                "mastery": "GAP",
                "evidence_level": "TEACHER_SCOPE",
                "evidence_confidence": 1,
                "frequency": 0.9,
                "prerequisite_ids": [],
            },
            {
                "id": "bayes",
                "course_id": "probability",
                "name": "贝叶斯公式",
                "exam_points": 12,
                "learning_minutes": 30,
                "mastery": "UNSEEN",
                "evidence_level": "PAST_EXAM",
                "evidence_confidence": 0.9,
                "frequency": 0.8,
                "prerequisite_ids": ["conditional"],
            },
        ],
    ).status_code == 200
    assert client.put(
        "/courses/probability/goals/final",
        json={
            "id": "final",
            "course_id": "probability",
            "exam_at": "2026-12-20T09:00:00+08:00",
            "available_minutes": 45,
            "target_score": 80,
        },
    ).status_code == 200

    response = client.post("/goals/final/plan")

    assert response.status_code == 200
    body = response.json()
    assert body["planned_minutes"] == 45
    assert [item["topic_id"] for item in body["items"] if item["tier"] == "MUST"] == [
        "conditional",
        "bayes",
    ]
    assert client.get("/goals/final/plan").json() == body
