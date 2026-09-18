from fastapi.testclient import TestClient

from studypilot.api import create_app


def test_parse_search_window_and_verify_citation_api(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db", tmp_path / "data")) as client:
        client.put("/courses/p", json={"id": "p", "name": "概率论"})
        uploaded = client.post(
            "/courses/p/sources",
            files={
                "file": (
                    "note.md",
                    "# 贝叶斯公式\n\n贝叶斯公式用于根据结果反推原因。".encode(),
                    "text/markdown",
                )
            },
            data={"document_kind": "PERSONAL_NOTE", "trust_level": "MEDIUM"},
        ).json()
        source_id = uploaded["asset"]["id"]
        parsed = client.post(
            f"/sources/{source_id}/parse", json={"parser_kind": "MARKDOWN"}
        )
        assert parsed.status_code == 200
        block = parsed.json()[0]
        assert client.get(f"/blocks/{block['id']}").json() == block

        hits = client.post("/courses/p/search", json={"query": "贝叶斯", "limit": 5})
        assert hits.status_code == 200
        citation = hits.json()[0]["citation"]
        assert client.post("/citations/verify", json=citation).json()["valid"] is True

        window = client.post(
            "/courses/p/knowledge-window",
            json={"query": "贝叶斯", "max_blocks": 1, "max_chars": 100},
        )
        assert window.status_code == 200
        assert len(window.json()["items"]) == 1

