#type: ignore
import sys
from types import SimpleNamespace as NS

import pytest
from fakes import FakeApprover, FakeModelClient
from NovaHarness.harness._prompts import BASE, MEMORY
from NovaHarness.harness.agent import MEMORY_TOOLS, Agent
from NovaHarness.harness.approval import CLIApprover
from NovaHarness.harness.memory import MemoryStore
from NovaHarness.harness.messages import ModelResponse, ToolCall
from NovaHarness.harness.policies import Policy
from NovaHarness.tools.tool import tool

DEFAULT_TOOLS = {
    "calculator",
    "current_time",
    "read_file",
    "list_directory",
    "find_files",
    "search_text",
    "write_file",
    "run_command",
    "fetch_url",
}


# --- helpers ---------------------------------------------------------------------


def say(text: str) -> ModelResponse:
    """A model turn that only answers."""
    return ModelResponse(response=text)


def use(name: str, args: dict, id: str = "c1") -> ModelResponse:
    """A model turn that requests one tool."""
    return ModelResponse(tool_calls=(ToolCall(tool_name=name, tool_id=id, tool_args=args),))


def call(name: str) -> ToolCall:
    return ToolCall(tool_name=name, tool_id="c1", tool_args={})


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"hello {name}"


@pytest.fixture
def store(tmp_path):
    """A memory store in tmp_path, so no test touches ~/.nova_harness."""
    return MemoryStore(tmp_path / "memory.db")


@pytest.fixture
def make(tmp_path, store):
    """Build an Agent with a scripted model, a yes-approver, and a temp workspace and store."""

    def _make(script=(), **kwargs) -> Agent:
        kwargs.setdefault("approver", FakeApprover(answer=True))
        kwargs.setdefault("memory", store)
        kwargs.setdefault("root", tmp_path)
        return Agent(FakeModelClient(script), **kwargs)

    return _make


# --- tools -----------------------------------------------------------------------


def test_defaults_register_default_and_memory_tools(make):
    assert set(make().tools) == DEFAULT_TOOLS | MEMORY_TOOLS


def test_web_search_is_added_when_a_search_fn_is_given(make):
    agent = make(search_fn=lambda query, n: [])
    assert "web_search" in agent.tools


def test_without_default_tools_only_memory_and_caller_tools_remain(make):
    agent = make(tools=[greet], use_default_tools=False)
    assert set(agent.tools) == MEMORY_TOOLS | {"greet"}


def test_memory_false_registers_no_memory_tools(make):
    agent = make(memory=False)
    assert agent.memory is None
    assert not MEMORY_TOOLS & set(agent.tools)


def test_caller_tool_is_added_next_to_the_defaults(make):
    agent = make(tools=[greet])
    assert set(agent.tools) == DEFAULT_TOOLS | MEMORY_TOOLS | {"greet"}


def test_caller_tool_replaces_a_default_with_the_same_name(make):
    @tool
    def read_file(path: str) -> str:
        """A replacement read_file."""
        return "replaced"

    agent = make(tools=[read_file])
    assert agent.tools["read_file"] is read_file


# --- policy ----------------------------------------------------------------------


def test_default_policy(make):
    policy = make(tools=[greet]).policy
    assert policy.evaluate(call("read_file")).action == "allow"
    assert policy.evaluate(call("write_file")).action == "ask"
    assert policy.evaluate(call("run_command")).action == "ask"
    for name in MEMORY_TOOLS:
        assert policy.evaluate(call(name)).action == "ask"
    assert policy.evaluate(call("greet")).action == "ask"  # no rule: safe default


def test_rules_override_only_the_named_tools(make):
    policy = make(policy={"write_file": "allow", "read_file": "deny"}).policy
    assert policy.evaluate(call("write_file")).action == "allow"
    assert policy.evaluate(call("read_file")).action == "deny"
    assert policy.evaluate(call("calculator")).action == "allow"  # untouched
    assert policy.evaluate(call("run_command")).action == "ask"  # untouched


def test_rules_can_loosen_the_memory_tools(make):
    policy = make(policy={"save_to_memory": "allow"}).policy
    assert policy.evaluate(call("save_to_memory")).action == "allow"


def test_empty_rules_keep_the_base_policy(make):
    assert make(policy={}).policy == make().policy


def test_a_policy_object_replaces_the_base(make):
    custom = Policy(allow=frozenset({"greet"}), default="deny")
    agent = make(tools=[greet], policy=custom)
    assert agent.policy is custom
    assert agent.policy.evaluate(call("read_file")).action == "deny"


def test_unknown_action_in_rules_raises(make):
    with pytest.raises(ValueError, match="write_file"):
        make(policy={"write_file": "Allow"})


# --- system prompt and memory ----------------------------------------------------


def test_conversation_starts_with_a_system_message(make):
    agent = make(system_prompt="You are a test.")
    assert len(agent.state) == 1
    first = agent.state.messages[0]
    assert first.role == "system"
    assert first.content == "You are a test."


def test_saved_memory_is_in_the_first_system_prompt(make, store):
    store.save("name", "Bakr")
    content = make(system_prompt="You are a test.").state.messages[0].content
    assert content.startswith("You are a test.")
    assert "name: Bakr" in content


def test_memory_false_keeps_memory_out_of_the_prompt(make, store):
    store.save("name", "Bakr")
    content = make(memory=False, system_prompt="You are a test.").state.messages[0].content
    assert content == "You are a test."


# --- running ---------------------------------------------------------------------


def test_run_returns_the_models_answer(make):
    assert make([say("hi there")]).run("hello") == "hi there"


def test_run_uses_an_allowed_tool_then_answers(make):
    agent = make([use("calculator", {"expression": "2 + 3"}), say("it is 5")])
    assert agent.run("what is 2 + 3?") == "it is 5"
    assert [m.role for m in agent.state.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.state.messages[3].content == "5"


def test_model_is_sent_every_registered_tool(make):
    agent = make([say("ok")], tools=[greet])
    agent.run("hi")
    _, specs = agent.model.calls[0]
    assert {s.name for s in specs} == set(agent.tools)


def test_ask_tools_go_through_the_approver(make, tmp_path):
    approver = FakeApprover(answer=True)
    agent = make(
        [use("write_file", {"path": "a.txt", "content": "hi"}), say("done")], approver=approver
    )
    assert agent.run("write a file") == "done"
    assert [c.tool_name for c, _ in approver.asked] == ["write_file"]
    assert (tmp_path / "a.txt").read_text() == "hi"


def test_denied_tool_does_not_run(make, tmp_path):
    agent = make(
        [use("write_file", {"path": "a.txt", "content": "hi"}), say("could not write")],
        approver=FakeApprover(answer=False),
    )
    assert agent.run("write a file") == "could not write"
    assert not (tmp_path / "a.txt").exists()
    assert agent.state.messages[3].content == "Denied. Do not retry."


def test_model_can_save_memory_for_the_next_conversation(make, store):
    agent = make([use("save_to_memory", {"key": "city", "value": "Rabat"}), say("noted")])
    agent.run("I live in Rabat")
    assert store.all("facts") == {"city": "Rabat"}
    agent.reset()
    assert "city: Rabat" in agent.state.messages[0].content


def test_history_is_kept_between_runs(make):
    agent = make([say("one"), say("two")], system_prompt="You are a test.")
    agent.run("first")
    agent.run("second")
    assert [m.role for m in agent.state.messages] == ["system", "user", "assistant", "user", "assistant"]
    second_call_messages, _ = agent.model.calls[1]
    assert second_call_messages[1].content == "first"


def test_reset_starts_a_new_conversation(make):
    agent = make([say("one")], system_prompt="You are a test.")
    agent.run("first")
    agent.reset()
    assert len(agent.state) == 1
    assert agent.state.messages[0].content == "You are a test."


# --- construction ----------------------------------------------------------------


def test_model_name_builds_a_litellm_client(monkeypatch, make):
    monkeypatch.setitem(sys.modules, "litellm", NS(completion=lambda **kw: None))
    agent = Agent("ollama/test", approver=FakeApprover(), memory=False, temperature=0.2)
    assert type(agent.model).__name__ == "LiteLLMClient"
    assert agent.model.model == "ollama/test"
    assert agent.model._params["temperature"] == 0.2


def test_default_approver_is_the_cli_approver(store, tmp_path):
    agent = Agent(FakeModelClient([]), memory=store, root=tmp_path)
    assert isinstance(agent._approver, CLIApprover)

def test_default_prompt_has_the_memory_section_only_with_memory(make):
    assert MEMORY in make().state.messages[0].content
    content = make(memory=False).state.messages[0].content
    assert content.startswith(BASE)
    assert MEMORY not in content


def test_instructions_are_added_to_the_default_prompt(make):
    content = make(instructions="Answer in French.").state.messages[0].content
    assert content.startswith(BASE)
    assert content.endswith("## Additional instructions\nAnswer in French.")


def test_system_prompt_replaces_the_default(make):
    content = make(system_prompt="Custom.", instructions="ignored").state.messages[0].content
    assert content == "Custom."
    
def test_empty_system_prompt_sends_no_system_message(make):
    assert len(make(system_prompt="", memory=False).state) == 0