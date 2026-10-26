#type: ignore
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest
from fakes import FakeApprover, FakeModelClient

from harness.approval import ToolApprover
from harness.errors import MaxStepsReachedError, ModelError, PolicyAbortError
from harness.executor import Executor
from harness.messages import ModelResponse, ToolCall
from harness.policies import Policy
from harness.runtime import Runtime
from harness.state import ConversationState
from tools.registry import ToolRegistry
from tools.tool import tool

DENY_READ = Policy(deny=frozenset({"read_file"}))
ASK_READ = Policy(ask=frozenset({"read_file"}))


# --- helpers ---------------------------------------------------------------------


def say(text: str) -> ModelResponse:
    """A model turn that only answers."""
    return ModelResponse(response=text)


def use(*calls: tuple[str, dict]) -> ModelResponse:
    """A model turn that requests tools. Ids are c1, c2, ... within the turn."""
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(tool_name=name, tool_id=f"c{i}", tool_args=args)
            for i, (name, args) in enumerate(calls, start=1)
        )
    )


def read(path: str) -> tuple[str, dict]:
    return ("read_file", {"path": path})


def tc(name: str = "read_file", args: dict | None = None, id: str = "c1") -> ToolCall:
    return ToolCall(
        tool_name=name,
        tool_id=id,
        tool_args={"path": "a.txt"} if args is None else args,
    )


def roles(state: ConversationState) -> list[str]:
    return [m.role for m in state.messages]


@dataclass
class Rig:
    runtime: Runtime
    model: FakeModelClient
    approver: Any
    executed: list[str] = field(default_factory=list)  # paths read_file actually ran with


def build(
    script: Iterable[ModelResponse | Exception] = (),
    *,
    policy: Policy | None = None,
    approver: Any = None,
    max_steps: int = 10,
    max_denials: int = 2,
) -> Rig:
    executed: list[str] = []

    @tool
    def read_file(path: str) -> str:
        """Read a file (fake)."""
        executed.append(path)
        return f"contents of {path}"

    @tool
    def explode() -> str:
        """Always fails."""
        raise ValueError("kaboom")

    registry = ToolRegistry([read_file, explode])
    model = FakeModelClient(script)
    approver = FakeApprover(True) if approver is None else approver
    runtime = Runtime(
        model=model,
        tool_registry=registry,
        executor=Executor(registry),
        policy=policy if policy is not None else Policy(allow=frozenset({"read_file", "explode"})),
        approver=approver,
        max_steps=max_steps,
        max_denials=max_denials,
    )
    return Rig(runtime, model, approver, executed)


# --- the loop ----------------------------------------------------------------------


def test_plain_answer_takes_one_model_call():
    rig = build([say("hi")])
    state = ConversationState()
    assert rig.runtime.run(state, "hello") == "hi"
    assert len(rig.model.calls) == 1
    assert roles(state) == ["user", "assistant"]


def test_tool_cycle_runs_each_requested_call_then_answers():
    rig = build([use(read("a.txt")), use(read("a.txt")), say("done")])
    assert rig.runtime.run(ConversationState(), "read a.txt twice") == "done"
    assert rig.executed == ["a.txt", "a.txt"]
    assert len(rig.model.calls) == 3


def test_history_shape_after_one_tool_turn():
    rig = build([use(read("a.txt")), say("done")])
    state = ConversationState()
    rig.runtime.run(state, "go")
    assert roles(state) == ["user", "assistant", "tool", "assistant"]
    assert state.messages[2].tool_call_id == "c1"
    assert state.messages[2].content == "contents of a.txt"


def test_parallel_calls_each_get_a_matching_result():
    rig = build([use(read("a"), read("b"), read("c")), say("done")])
    state = ConversationState()
    rig.runtime.run(state, "go")
    tool_messages = [m for m in state.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2", "c3"]
    assert [m.content for m in tool_messages] == [
        "contents of a",
        "contents of b",
        "contents of c",
    ]
    assert rig.executed == ["a", "b", "c"]


def test_history_carries_over_between_runs_on_the_same_state():
    rig = build([say("one"), say("two")])
    state = ConversationState()
    rig.runtime.run(state, "first")
    rig.runtime.run(state, "second")
    seen_by_second_call = rig.model.calls[1][0]
    assert [m.role for m in seen_by_second_call] == ["user", "assistant", "user"]


def test_model_receives_the_registered_tool_specs():
    rig = build([say("hi")])
    rig.runtime.run(ConversationState(), "hi")
    _, specs = rig.model.calls[0]
    assert {s.name for s in specs} == {"read_file", "explode"}


# --- policy: deny ---------------------------------------------------------------------


def test_denied_call_is_not_executed_and_the_model_gets_a_fixed_message():
    rig = build([use(read("a.txt")), say("ok")], policy=DENY_READ)
    state = ConversationState()
    assert rig.runtime.run(state, "go") == "ok"
    assert rig.executed == []
    assert state.messages[2].content == "Denied. Do not retry."


def test_denial_never_reveals_which_rule_fired():
    rig = build(policy=DENY_READ)
    result = rig.runtime.handle(tc(), {})
    assert result.status == "denied"
    assert result.tool_result == "Denied. Do not retry."  # nothing from Decision.reason


# --- policy: ask -----------------------------------------------------------------------


def test_ask_approved_runs_the_tool():
    approver = FakeApprover(True)
    rig = build(policy=ASK_READ, approver=approver)
    result = rig.runtime.handle(tc(), {})
    assert result.status == "ok"
    assert rig.executed == ["a.txt"]
    asked_call, reason = approver.asked[0]
    assert asked_call.tool_name == "read_file"
    assert reason  # the human is told why; the model never is


def test_ask_refused_is_a_denial():
    rig = build(policy=ASK_READ, approver=FakeApprover(False))
    result = rig.runtime.handle(tc(), {})
    assert result.status == "denied"
    assert rig.executed == []


def test_an_approver_that_crashes_fails_closed():
    class CrashingApprover:
        def approve(self, call: ToolCall, reason: str | None = None) -> bool:
            raise RuntimeError("UI crashed")

    rig = build(policy=ASK_READ, approver=CrashingApprover())
    result = rig.runtime.handle(tc(), {})
    assert result.status == "denied"
    assert rig.executed == []


def test_default_approver_denies_every_ask():
    rig = build(policy=ASK_READ, approver=ToolApprover())
    assert rig.runtime.handle(tc(), {}).status == "denied"
    assert rig.executed == []


# --- unknown tools -------------------------------------------------------------------------


def test_unknown_tool_is_an_error_result_listing_available_names():
    rig = build(policy=Policy())  # default action is "ask"
    result = rig.runtime.handle(tc(name="read_fil"), {})
    assert result.status == "error"
    assert "read_file" in result.tool_result
    assert rig.approver.asked == []  # nobody is asked to approve a typo


# --- the retry guard ------------------------------------------------------------------------


def test_repeating_a_denied_call_aborts_the_run():
    rig = build([use(read("a")), use(read("a")), say("never")], policy=DENY_READ, max_denials=2)
    with pytest.raises(PolicyAbortError):
        rig.runtime.run(ConversationState(), "go")
    assert len(rig.model.calls) == 2


def test_denials_of_different_calls_do_not_add_up():
    rig = build([use(read("a")), use(read("b")), say("ok")], policy=DENY_READ, max_denials=2)
    assert rig.runtime.run(ConversationState(), "go") == "ok"


def test_argument_order_does_not_hide_a_repeat():
    rig = build(policy=DENY_READ, max_denials=2)
    denials: dict = {}
    rig.runtime.handle(tc(args={"a": 1, "b": 2}), denials)
    with pytest.raises(PolicyAbortError):
        rig.runtime.handle(tc(args={"b": 2, "a": 1}), denials)


def test_a_refused_ask_counts_toward_the_abort_limit():
    rig = build(policy=ASK_READ, approver=FakeApprover(False), max_denials=2)
    denials: dict = {}
    rig.runtime.handle(tc(), denials)
    with pytest.raises(PolicyAbortError):
        rig.runtime.handle(tc(), denials)


def test_abort_leaves_the_last_assistant_turn_unanswered():
    # Known v0 rough edge, pinned on purpose: after PolicyAbortError the history ends
    # with tool calls that have no results. Providers reject such a history, so the
    # Agent must reset or trim the state before reusing it.
    rig = build([use(read("a")), use(read("a"))], policy=DENY_READ, max_denials=2)
    state = ConversationState()
    with pytest.raises(PolicyAbortError):
        rig.runtime.run(state, "go")
    last = state.messages[-1]
    assert last.role == "assistant" and last.tool_calls


# --- tool problems become results, not crashes ------------------------------------------------


def test_a_crashing_tool_becomes_an_error_result_and_the_run_continues():
    rig = build([use(("explode", {})), say("recovered")])
    state = ConversationState()
    assert rig.runtime.run(state, "go") == "recovered"
    assert "explode" in state.messages[2].content
    assert "ValueError" in state.messages[2].content


@pytest.mark.parametrize(
    "bad_args",
    [{}, {"path": ["not", "a", "string"]}, {"path": "a.txt", "extra": 1}],
    ids=["missing", "wrong-type", "unknown-arg"],
)
def test_invalid_arguments_become_an_error_result(bad_args):
    rig = build()
    result = rig.runtime.handle(tc(args=bad_args), {})
    assert result.status == "error"
    assert rig.executed == []


def test_the_run_survives_invalid_arguments():
    rig = build([use(("read_file", {})), say("fixed it")])
    assert rig.runtime.run(ConversationState(), "go") == "fixed it"


# --- limits and failures ---------------------------------------------------------------------------


def test_max_steps_stops_a_run_that_never_answers():
    rig = build([use(read("a"))] * 3, max_steps=3)
    with pytest.raises(MaxStepsReachedError):
        rig.runtime.run(ConversationState(), "go")
    assert len(rig.model.calls) == 3  # no fourth call


def test_model_errors_propagate_to_the_caller():
    rig = build([ModelError("rate limited")])
    with pytest.raises(ModelError):
        rig.runtime.run(ConversationState(), "go")