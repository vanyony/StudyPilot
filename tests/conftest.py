"""Old component tests explicitly use offline decisions; product never defaults to fake."""
import pytest
from agent_fixtures import RegressionLearningProvider
from studypilot.application.teaching import TeachingWorkflow, DeterministicFakeEvaluator
from studypilot.application.teaching_service import TeachingSessionService


@pytest.fixture(autouse=True)
def offline_session_regression(request, monkeypatch):
    if request.path.name not in {"test_teaching.py", "test_teaching_api.py", "test_teaching_cas.py", "test_channel.py", "test_web.py", "test_llm_providers.py", "test_api.py"}:
        return
    original_workflow = TeachingWorkflow.__init__
    original_service = TeachingSessionService.__init__

    def workflow_init(self, *args, **kwargs):
        kwargs.setdefault("learning_provider", RegressionLearningProvider(kwargs.get("teacher_provider")))
        if kwargs.get("learning_provider") is None:
            kwargs["learning_provider"] = RegressionLearningProvider(kwargs.get("teacher_provider"))
        if kwargs.get("evaluator") is None:
            kwargs["evaluator"] = DeterministicFakeEvaluator()
        original_workflow(self, *args, **kwargs)

    def service_init(self, *args, **kwargs):
        kwargs.setdefault("learning_provider", RegressionLearningProvider(kwargs.get("teacher_provider")))
        if kwargs.get("learning_provider") is None:
            kwargs["learning_provider"] = RegressionLearningProvider(kwargs.get("teacher_provider"))
        if kwargs.get("evaluator") is None:
            kwargs["evaluator"] = DeterministicFakeEvaluator()
        original_service(self, *args, **kwargs)

    monkeypatch.setattr(TeachingWorkflow, "__init__", workflow_init)
    monkeypatch.setattr(TeachingSessionService, "__init__", service_init)