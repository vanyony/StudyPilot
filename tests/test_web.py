from __future__ import annotations

from fastapi.testclient import TestClient

from studypilot.api import create_app


def _prepare(client: TestClient) -> None:
    assert client.post(
        "/study/courses",
        data={"course_id": "course", "name": "概率论"},
        follow_redirects=False,
    ).status_code == 303
    assert client.post(
        "/study/courses/course/goals",
        data={
            "goal_id": "final",
            "exam_at": "2026-12-20T09:00",
            "available_minutes": "30",
        },
        follow_redirects=False,
    ).status_code == 303
    assert client.post(
        "/study/courses/course/topics",
        data={
            "topics_text": "bayes|贝叶斯公式|10|10|UNSEEN|",
        },
        follow_redirects=False,
    ).status_code == 303


def test_study_page_renders_provider_boundary_and_empty_workspace(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db")) as client:
        response = client.get("/study")

    assert response.status_code == 200
    assert "本地学习入口" in response.text
    assert "Deterministic demo（默认）" in response.text
    assert "未配置 key/model" in response.text
    assert "保存课程" in response.text


def test_web_forms_connect_course_goal_topics_plan_and_sources(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db", tmp_path / "data")) as client:
        _prepare(client)
        assert client.post(
            "/study/courses/course/plan",
            data={"goal_id": "final"},
            follow_redirects=False,
        ).status_code == 303

        uploaded = client.post(
            "/study/courses/course/sources",
            files={"file": ("notes.txt", b"# Bayes\nposterior probability", "text/plain")},
            data={"document_kind": "COURSE_MATERIAL", "trust_level": "HIGH"},
            follow_redirects=False,
        )
        assert uploaded.status_code == 303
        source = client.app.state.repository.list_sources("course")[0]
        assert client.post(
            f"/study/sources/{source.asset.id}/parse",
            data={"parser_kind": "MARKDOWN"},
            follow_redirects=False,
        ).status_code == 303

        page = client.get("/study?course_id=course&goal_id=final")

    assert page.status_code == 200
    assert "必学" in page.text
    assert "notes.txt" in page.text
    assert "解析" in page.text


def test_web_session_shows_question_feedback_and_disabled_submit_script(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db", tmp_path / "data")) as client:
        _prepare(client)
        uploaded = client.post(
            "/study/courses/course/sources",
            files={"file": ("teaching-notes.txt", "# 贝叶斯公式\nposterior probability".encode(), "text/plain")},
            data={"document_kind": "COURSE_MATERIAL", "trust_level": "HIGH"},
            follow_redirects=False,
        )
        assert uploaded.status_code == 303
        source = client.app.state.repository.list_sources("course")[0]
        assert client.post(
            f"/study/sources/{source.asset.id}/parse",
            data={"parser_kind": "MARKDOWN"},
            follow_redirects=False,
        ).status_code == 303
        started = client.post(
            "/study/courses/course/teaching-sessions",
            data={"goal_id": "final", "session_id": "web-session", "start": "true"},
            follow_redirects=False,
        )
        assert started.status_code == 303
        state = client.app.state.teaching_service.get_state("web-session")
        assert state is not None
        waiting = client.get(
            "/study?course_id=course&goal_id=final&session_id=web-session"
        )
        assert waiting.status_code == 200
        assert "WAITING_ANSWER" in waiting.text
        assert "提交答案" in waiting.text
        assert "讲解" in waiting.text
        assert "Knowledge Window 引用" in waiting.text
        assert "teaching-notes.txt" in waiting.text
        assert "posterior probability" in waiting.text
        assert "原文" in waiting.text

        answered = client.post(
            "/study/sessions/web-session/answer",
            data={
                "message_id": "web-answer-1",
                "expected_version": str(state.version),
                "answer": "correct",
            },
            follow_redirects=False,
        )
        assert answered.status_code == 303
        feedback = client.get(answered.headers["location"])
        script = client.get("/static/study.js")

    assert feedback.status_code == 200
    assert "评分反馈" in feedback.text
    assert "READY" in feedback.text
    assert "提交中" in script.text
    assert "disabled" in script.text


def test_web_errors_are_readable_and_restart_keeps_session_state(tmp_path) -> None:
    database = tmp_path / "study.db"
    with TestClient(create_app(database)) as first:
        _prepare(first)
        started = first.post(
            "/study/courses/course/teaching-sessions",
            data={"goal_id": "final", "session_id": "restart-web", "start": "true"},
            follow_redirects=False,
        )
        assert started.status_code == 303
        state = first.app.state.teaching_service.get_state("restart-web")
        assert state is not None

    with TestClient(create_app(database)) as restarted:
        page = restarted.get(
            "/study?course_id=course&goal_id=final&session_id=restart-web"
        )
        stale = restarted.post(
            "/study/sessions/restart-web/answer",
            data={
                "message_id": "stale-web",
                "expected_version": str(state.version + 1),
                "answer": "wrong",
            },
            follow_redirects=False,
        )
        error_page = restarted.get(stale.headers["location"])

    assert page.status_code == 200
    assert "WAITING_ANSWER" in page.text
    assert stale.status_code == 303
    assert "[409]" in error_page.text
    assert "does not match expected version" in error_page.text


def test_web_invalid_topic_form_returns_422_flash(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "study.db")) as client:
        _prepare(client)
        response = client.post(
            "/study/courses/course/topics",
            data={"topics_text": "not-a-valid-topic-line"},
            follow_redirects=False,
        )
        page = client.get(response.headers["location"])

    assert response.status_code == 303
    assert "[422]" in page.text
    assert "格式错误" in page.text
