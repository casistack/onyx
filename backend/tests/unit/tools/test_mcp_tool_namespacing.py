from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.interface import Tool
from onyx.tools.models import ToolCallKickoff
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.mcp.mcp_tool import MCPTool
from onyx.tools.tool_runner import run_tool_calls


class DummyTool(Tool[None]):
    def __init__(
        self,
        *,
        tool_id: int,
        name: str,
        llm_name: str,
        emitter: Emitter,
    ) -> None:
        super().__init__(emitter=emitter)
        self._id = tool_id
        self._name = name
        self._llm_name = llm_name
        self.run_mock = MagicMock(
            return_value=ToolResponse(
                rich_response=None,
                llm_facing_response=f"{llm_name}-ok",
            )
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def llm_name(self) -> str:
        return self._llm_name

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def display_name(self) -> str:
        return self._name

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def emit_start(self, placement: Placement) -> None:
        del placement
        return None

    def run(
        self,
        placement: Placement,
        override_kwargs: None = None,
        **llm_kwargs: object,
    ) -> ToolResponse:
        return self.run_mock(
            placement=placement,
            override_kwargs=override_kwargs,
            **llm_kwargs,
        )


def test_mcp_tool_definition_uses_namespaced_llm_name() -> None:
    emitter = Emitter(Queue())
    mcp_server = SimpleNamespace(name="graphiti")
    mcp_tool = MCPTool(
        tool_id=1,
        emitter=emitter,
        mcp_server=mcp_server,
        tool_name="add_memory",
        tool_description="Persist memory",
        tool_definition={"type": "object", "properties": {}},
    )

    definition = mcp_tool.tool_definition()

    assert definition["function"]["name"] == "mcp_graphiti_add_memory"
    assert mcp_tool.name == "add_memory"


def test_run_tool_calls_routes_using_llm_name_for_duplicate_canonical_names() -> None:
    emitter = Emitter(Queue())
    built_in_memory_tool = DummyTool(
        tool_id=1,
        name="add_memory",
        llm_name="add_memory",
        emitter=emitter,
    )
    mcp_memory_tool = DummyTool(
        tool_id=2,
        name="add_memory",
        llm_name="mcp_graphiti_add_memory",
        emitter=emitter,
    )

    result = run_tool_calls(
        tool_calls=[
            ToolCallKickoff(
                tool_call_id="call_1",
                tool_name="mcp_graphiti_add_memory",
                tool_args={"episode_body": "hello"},
                placement=Placement(turn_index=0, tab_index=0),
            )
        ],
        tools=[built_in_memory_tool, mcp_memory_tool],
        message_history=[
            ChatMessageSimple(
                message="hello",
                token_count=1,
                message_type=MessageType.USER,
            )
        ],
        user_memory_context=None,
        user_info=None,
        citation_mapping={},
        next_citation_num=1,
    )

    assert len(result.tool_responses) == 1
    assert result.tool_responses[0].llm_facing_response == "mcp_graphiti_add_memory-ok"
    built_in_memory_tool.run_mock.assert_not_called()
    mcp_memory_tool.run_mock.assert_called_once()
