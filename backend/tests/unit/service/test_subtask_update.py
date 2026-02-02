
import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from app.service.chat_service import step_solve
from app.model.chat import Chat, UpdateData
from app.service.task import (
    TaskLock,
    Action,
    ActionImproveData,
    ActionUpdateTaskData,
    task_locks
)
from camel.tasks import Task

def parse_sse(sse_string):
    """Parse SSE string 'data: {...}\n\n' into dict"""
    if sse_string.startswith("data: "):
        json_str = sse_string[6:].strip()
        return json.loads(json_str)
    return json.loads(sse_string)

@pytest.mark.asyncio
async def test_update_task_persistence(temp_dir):
    project_id = "test_project"

    options = Chat(
        project_id=project_id,
        task_id="test_task",
        email="test@example.com",
        question="Initial question",
        model_platform="openai",
        model_type="gpt-4",
        api_key="test",
        api_url="http://test",
        env_path=str(temp_dir / ".env")
    )

    request = AsyncMock()
    request.is_disconnected.return_value = False

    queue = asyncio.Queue()
    task_lock = TaskLock(id=project_id, queue=queue, human_input={})
    task_locks[project_id] = task_lock

    # Put initial item in queue to start step_solve
    await task_lock.put_queue(ActionImproveData(data="Initial question"))

    try:
        mock_agent = MagicMock()
        mock_summary_task = AsyncMock(return_value="Task Summary")

        with patch("app.service.chat_service.question_confirm", return_value=True), \
             patch("app.service.chat_service.construct_workforce") as mock_construct, \
             patch("app.service.chat_service.task_summary_agent"), \
             patch("app.service.chat_service.question_confirm_agent", return_value=mock_agent), \
             patch("app.service.chat_service.summary_task", mock_summary_task):

            mock_workforce = MagicMock()

            # Mock side effect to update camel_task.subtasks
            def mock_make_sub_tasks(task, *args, **kwargs):
                subtasks = [
                    Task(content="Subtask 1", id="1.1"),
                    Task(content="Subtask 2", id="1.2")
                ]
                # Update the passed task object
                task.subtasks = list(subtasks)
                return subtasks

            mock_workforce.eigent_make_sub_tasks.side_effect = mock_make_sub_tasks
            mock_construct.return_value = (mock_workforce, MagicMock())

            print("Starting generator")
            generator = step_solve(options, request, task_lock)

            print("Waiting for confirmed")
            item_sse = await generator.__anext__()
            item = parse_sse(item_sse)
            print(f"Got item: {item['step']}")
            assert item["step"] == "confirmed"

            print("Waiting for background task")
            # Loop a bit to let background task run and put item in queue
            for _ in range(20):
                await asyncio.sleep(0.1)
                if not queue.empty():
                    print("Queue has item")
                    break

            print("Waiting for to_sub_tasks")
            item_sse = await generator.__anext__()
            item = parse_sse(item_sse)
            print(f"Got item: {item['step']}")
            assert item["step"] == "to_sub_tasks"

            assert hasattr(task_lock, "decompose_sub_tasks")
            initial_subtasks = task_lock.decompose_sub_tasks
            print(f"Initial subtasks: {len(initial_subtasks)}")
            assert len(initial_subtasks) == 2

            update_payload = UpdateData(task=[
                {"id": "1.2", "content": "Subtask 2 Updated"},
                {"id": "", "content": "Subtask 3 New"}
            ])

            print("Sending update")
            await task_lock.put_queue(ActionUpdateTaskData(data=update_payload))

            print("Waiting for update response")
            item_sse = await generator.__anext__()
            item = parse_sse(item_sse)
            print(f"Got item: {item['step']}")
            assert item["step"] == "to_sub_tasks"

            sub_tasks_data = item["data"]["sub_tasks"]
            ids = [t["id"] for t in sub_tasks_data]
            contents = {t["id"]: t["content"] for t in sub_tasks_data}

            print(f"Result IDs: {ids}")
            assert "1.1" not in ids
            assert "1.2" in ids
            assert contents["1.2"] == "Subtask 2 Updated"

            new_id = [i for i in ids if i != "1.2"][0]
            # Since max ID was 1.2, new ID should be 1.3
            assert new_id.endswith("3")
            assert contents[new_id] == "Subtask 3 New"

            persisted_subtasks = task_lock.decompose_sub_tasks
            persisted_ids = [t.id for t in persisted_subtasks]
            assert "1.1" not in persisted_ids
            assert "1.2" in persisted_ids
            assert new_id in persisted_ids

            await generator.aclose()

    finally:
        if project_id in task_locks:
            del task_locks[project_id]
