import time
from harness.approval import CLIApprover
from harness.messages import ToolCall
from harness.approval import CLIApprover

def test_late_answer_never_approves_the_next_question():
    script = iter([(0.3, "y"), (0.0, "n")])   # (delay, answer) per read

    def scripted(prompt: str) -> str:
        delay, answer = next(script)
        time.sleep(delay)
        return answer

    approver = CLIApprover(timeout=0.1, input_fn=scripted)
    call = ToolCall("c1", "write_file", {})

    assert approver.approve(call, reason=None) is False   # times out; "y" arrives late
    time.sleep(0.4)                           # late "y" is now in the queue
    assert approver.approve(call, reason=None) is False   # must read "n", not the stale "y"