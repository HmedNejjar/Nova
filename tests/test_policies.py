from harness.messages import ToolCall
from harness.policies import Policy


def call(name: str) -> ToolCall:
    return ToolCall(tool_id="c1", tool_name=name, tool_args={})


def test_allow_list():
    policy = Policy(allow=frozenset({"read_file"}))
    assert policy.evaluate(call("read_file")).action == "allow"


def test_ask_list():
    policy = Policy(ask=frozenset({"write_file"}))
    assert policy.evaluate(call("write_file")).action == "ask"


def test_deny_list():
    policy = Policy(deny=frozenset({"delete_file"}))
    assert policy.evaluate(call("delete_file")).action == "deny"


def test_unlisted_tool_gets_default():
    assert Policy().evaluate(call("anything")).action == "ask"
    assert Policy(default="deny").evaluate(call("anything")).action == "deny"


def test_strictest_wins_when_tool_is_in_several_lists():
    both = frozenset({"x"})
    assert Policy(allow=both, ask=both).evaluate(call("x")).action == "ask"
    assert Policy(allow=both, deny=both).evaluate(call("x")).action == "deny"
    assert Policy(ask=both, deny=both).evaluate(call("x")).action == "deny"
    assert Policy(allow=both, ask=both, deny=both).evaluate(call("x")).action == "deny"


def test_listed_tool_ignores_default():
    # An allow-listed tool is allowed even when the default is deny.
    policy = Policy(allow=frozenset({"read_file"}), default="deny")
    assert policy.evaluate(call("read_file")).action == "allow"