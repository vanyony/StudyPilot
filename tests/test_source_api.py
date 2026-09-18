from fastapi.testclient import TestClient

from studypilot.api import create_app


def test_upload_list_and_read_source(tmp_path) -> None:
    database = tmp_path / "api.db"
    storage = tmp_path / "data"
    app = create_app(database, storage)
    with TestClient(app) as client:
        client.put("/courses/probability", json={"id": "probability", "name": "概率论"})
        response = client.post(
            "/courses/probability/sources",
            files={"file": ("exam.pdf", b"pdf bytes", "application/pdf")},
            data={
                "document_kind": "EXAM_PAPER",
                "trust_level": "HIGH",
                "origin": "teacher",
                "metadata": '{"year":2024}',
            },
        )
        assert response.status_code == 201
        source_id = response.json()["asset"]["id"]
        assert response.json()["blob"]["parse_status"] == "PENDING"
        assert len(client.get("/courses/probability/sources").json()) == 1
        assert client.get(f"/sources/{source_id}").json() == response.json()

    # A fresh app instance proves the API reads persisted state after restart.
    with TestClient(create_app(database, storage)) as restarted:
        assert restarted.get(f"/sources/{source_id}").status_code == 200


def test_invalid_source_enums_return_422(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api.db", tmp_path / "data")) as client:
        client.put("/courses/c", json={"id": "c", "name": "course"})
        response = client.post(
            "/courses/c/sources",
            files={"file": ("note.md", b"note", "text/markdown")},
            data={"document_kind": "UNKNOWN", "trust_level": "HIGH"},
        )

    assert response.status_code == 422

