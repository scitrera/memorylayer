"""End-to-end connector normalization through the ordinary memory API."""

from fastapi.testclient import TestClient


def test_connector_envelope_is_normalized_persisted_and_recalled(test_client: TestClient) -> None:
    headers = {"X-Workspace-ID": "knowledge_work_connector_api"}
    created = test_client.post(
        "/v1/memories",
        headers=headers,
        json={
            "content": "Work register QAT-917 assigns the audit preparation to Imani Reed.",
            "metadata": {
                "connector_type": "jira",
                "connector_record": {
                    "record_type": "issue",
                    "key": "QAT-917",
                    "fields": {
                        "summary": "Quasar Audit Task",
                        "assignee": {"displayName": "Imani Reed", "accountId": "im-22"},
                        "project": {"name": "Quasar Compliance", "key": "QCP"},
                    },
                },
            },
        },
    )
    assert created.status_code == 201, created.text
    memory = created.json()["memory"]
    assert memory["metadata"]["knowledge_work"]["subject"]["name"] == "Quasar Audit Task"
    assert memory["relation_write_result"]["resolved"] == 2

    recalled = test_client.post(
        "/v1/memories/recall",
        headers=headers,
        json={
            "query": "Who is assigned to Quasar Audit Task?",
            "limit": 10,
            "include_associations": False,
            "traverse_depth": 0,
        },
    )
    assert recalled.status_code == 200, recalled.text
    body = recalled.json()
    assert memory["id"] in {item["id"] for item in body["memories"]}
    assert body["relation_paths"]
