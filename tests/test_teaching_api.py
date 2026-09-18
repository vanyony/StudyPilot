from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from studypilot.api import create_app
from studypilot.application.teaching import DeterministicFakeEvaluator


class CountingEvaluator(DeterministicFakeEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return super().evaluate(*args, **kwargs)


def _prepare(client: TestClient) -> None:
    assert client.put("/courses/c", json={"id": "c", "name": "course"}).status_code == 200
    assert client.put(
        "/courses/c/topics",
        json=[
            {
                "id": "topic-a",
                "course_id": "c",
                "name": "topic-a",
                "exam_points": 10,
                "learning_minutes": 10,
                "mastery": "UNSEEN",
                "evidence_level": "PAST_EXAM",
                "evidence_confidence": 1,
                "frequency": 0.8,
                "prerequisite_ids": [],
            }
        ],
    ).status_code == 200
    assert client.put(
        "/courses/c/goals/g",
        json={
            "id": "g",
            "course_id": "c",
            "exam_at": "2026-12-20T09:00:00+08:00",
            "available_minutes": 30,
        },
    ).status_code == 200


def _create_and_start(client: TestClient) -> dict:
    created = client.post(
        "/courses/c/teaching-sessions",
        json={"goal_id": "g", "session_id": "stable-session"},
    )
    assert created.status_code == 201
    assert created.json()["status"] == "CREATED"
    started = client.post("/teaching-sessions/stable-session/start")
    assert started.status_code == 200
    state = started.json()
    assert state["thread_id"] == "stable-session"
    assert state["status"] == "WAITING_ANSWER"
    return state


def test_create_start_read_and_answer_api(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db")) as client:
        _prepare(client)
        state = _create_and_start(client)

        fetched = client.get("/teaching-sessions/stable-session")
        assert fetched.status_code == 200
        assert fetched.json() == state

        answered = client.post(
            "/teaching-sessions/stable-session/answer",
            json={
                "message_id": "m-1",
                "expected_version": state["version"],
                "answer": "correct",
            },
        )
        assert answered.status_code == 200
        assert answered.json()["status"] == "COMPLETED"
        assert answered.json()["mastery_state"] == "READY"


def test_duplicate_message_returns_original_result_without_rescoring(tmp_path) -> None:
    evaluator = CountingEvaluator()
    with TestClient(create_app(tmp_path / "study.db", evaluator=evaluator)) as client:
        _prepare(client)
        state = _create_and_start(client)
        payload = {
            "message_id": "m-1",
            "expected_version": state["version"],
            "answer": "wrong",
        }
        first = client.post("/teaching-sessions/stable-session/answer", json=payload)
        duplicate = client.post("/teaching-sessions/stable-session/answer", json=payload)

        assert first.status_code == duplicate.status_code == 200
        assert duplicate.json() == first.json()
        assert evaluator.calls == 1

        changed = client.post(
            "/teaching-sessions/stable-session/answer",
            json={**payload, "answer": "correct"},
        )
        assert changed.status_code == 409


def test_same_session_operations_are_serialized_in_one_process(tmp_path) -> None:
    evaluator = CountingEvaluator()
    with TestClient(create_app(tmp_path / "study.db", evaluator=evaluator)) as client:
        _prepare(client)
        state = _create_and_start(client)
        service = client.app.state.teaching_service

        def submit():
            return service.submit_answer(
                "stable-session",
                message_id="concurrent-message",
                answer="wrong",
                expected_version=state["version"],
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: submit(), range(2)))

        assert results[0] == results[1]
        assert evaluator.calls == 1


def test_stale_version_and_non_waiting_states_are_rejected(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db")) as client:
        _prepare(client)
        created = client.post(
            "/courses/c/teaching-sessions", json={"goal_id": "g", "session_id": "s"}
        )
        assert created.status_code == 201
        not_started = client.post(
            "/teaching-sessions/s/answer",
            json={"message_id": "m-0", "expected_version": 0, "answer": "correct"},
        )
        assert not_started.status_code == 409

        state = client.post("/teaching-sessions/s/start").json()
        stale = client.post(
            "/teaching-sessions/s/answer",
            json={"message_id": "m-1", "expected_version": state["version"] + 1, "answer": "wrong"},
        )
        assert stale.status_code == 409

        completed = client.post(
            "/teaching-sessions/s/answer",
            json={"message_id": "m-2", "expected_version": state["version"], "answer": "correct"},
        )
        assert completed.status_code == 200
        after_completion = client.post(
            "/teaching-sessions/s/answer",
            json={"message_id": "m-3", "expected_version": completed.json()["version"], "answer": "wrong"},
        )
        assert after_completion.status_code == 409

        assert client.get("/teaching-sessions/does-not-exist").status_code == 404


def test_missing_message_or_version_is_validation_error(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db")) as client:
        _prepare(client)
        _create_and_start(client)

        missing_both = client.post(
            "/teaching-sessions/stable-session/answer", json={"answer": "correct"}
        )
        assert missing_both.status_code == 422
        missing_version = client.post(
            "/teaching-sessions/stable-session/answer",
            json={"message_id": "m-1", "answer": "correct"},
        )
        assert missing_version.status_code == 422


def test_restart_reads_waiting_state_and_continues(tmp_path) -> None:
    database = tmp_path / "study.db"
    with TestClient(create_app(database)) as client:
        _prepare(client)
        state = _create_and_start(client)

    with TestClient(create_app(database)) as restarted:
        loaded = restarted.get("/teaching-sessions/stable-session")
        assert loaded.status_code == 200
        assert loaded.json() == state
        answered = restarted.post(
            "/teaching-sessions/stable-session/answer",
            json={
                "message_id": "after-restart",
                "expected_version": state["version"],
                "answer": "correct",
            },
        )
        assert answered.status_code == 200
        assert answered.json()["status"] == "COMPLETED"
