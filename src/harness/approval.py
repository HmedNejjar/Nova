from typing import Callable
from time import time
from math import ceil
from os import name as os_name
from sys import stdin, stdout
import json
from queue import Queue, Empty as QueueEmpty
from threading import Thread
from .messages import ToolCall

# ANSI escape codes: save cursor, restore cursor, clear the current line.
_SAVE, _RESTORE, _CLEAR = "\x1b7", "\x1b8", "\x1b[2K"

def _parse_answer(text: str) -> bool:
    """Parse the user's answer to a boolean.
    Accepts 'yes', 'y', 'true' (case-insensitive) as True, anything else as False.
    """
    return True if text.lower().strip() in ('yes', 'y', 'true') else False

class ToolApprover:
    """Decides whether one ToolCall may run."""
    
    def __init__(self, timeout: float = 3600.0, input_fn: Callable = input) -> None:
        self.timeout = timeout
        self.input_fn = input_fn
        self.default = False
    
    def approve(self, tool_call: ToolCall, reason: str | None) -> bool:
        """Returns True if the ToolCall is approved by user, False otherwise."""
        return self.default
    
    def describe(self, tool_call: ToolCall, reason: str | None = None, max_len: int = 200) -> str:
        """Return a human-readable description of the approval decision."""
        
        lines = [f"Tool '{tool_call.tool_name}' requires approval."]
        if reason:
            lines.append(f"  reason: {reason}")
        if not tool_call.tool_args:
            lines.append("  args:   (none)")

        # Cut each value, not the whole JSON: a value of up to max_len chars is shown in full,
        # so what the human approves is exactly what runs. json.dumps escapes newlines, so a
        # value can't fake extra lines in the prompt.
        for name, value in tool_call.tool_args.items():
            raw = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
            text = json.dumps(raw[:max_len], ensure_ascii=False)
            if len(raw) > max_len:
                text += f" ... [truncated: {len(raw):,} chars total]"
            lines.append(f"  {name}: {text}")
        return "\n".join(lines)
        
class CLIApprover(ToolApprover):
    """Prompts the user to approve or deny a ToolCall via the command line.
    
    Layout while waiting:
 
        Tool 'write_file' requires approval.
          args:   {"path": "notes.md"}
          42s left to answer            <- rewritten in place every second
        Approve? [y/n]: _               <- the human types here"""
    
    QUESTION = "Approve? [y/n]: "
    
    def __init__(self, timeout: float = 3600.0, input_fn: Callable = input, show_countdown: bool | None= None) -> None:
        super().__init__()
        self.timeout = timeout
        self.input_fn = input_fn
        
        # Whether to show a countdown.    
        if show_countdown is None:
            show_countdown = input_fn is input and stdin.isatty() and stdout.isatty()
        self._live = show_countdown
    
        if self._live:
            self._enable_ansi()
            
        # Requests to read a line from stdin.
        self._requests: Queue[None] = Queue()
        # Lines read from stdin.
        self._answers: Queue[str | None] = Queue()
        # Whether a read was requested and hasn't answered yet.
        self._pending = False
        # The thread that reads lines from stdin.
        self._reader: Thread | None = None
        
    @staticmethod
    def _enable_ansi() -> None:
        """Windows consoles need VT processing switched on before escape codes work."""
        if os_name != "nt":
            return
        try:
            import ctypes
    
            kernel32 = getattr(ctypes, "windll").kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass  # worst case: the countdown prints raw escape codes; approval still works
            
    def approve(self, tool_call: ToolCall, reason: str | None) -> bool:
        self._show(tool_call, reason)
        answer = self._read_line()
        approved = answer is not None and _parse_answer(answer)
        self._finish(answer, approved)
        return approved

    @staticmethod
    def _countdown(remaining: float) -> str:
        return f"  {ceil(remaining)}s left to answer"
 
    @staticmethod
    def _render(text: str, up: int) -> None:
        """Rewrite the countdown line in place, then put the cursor back where the human is typing."""
        # end="": a newline after _RESTORE would move the cursor off the answer line.
        print(f"{_SAVE}\x1b[{up}A\r{_CLEAR}{text}{_RESTORE}", end="", flush=True)
        
    def _read_line(self) -> str | None:
        """One line from the human, or None on timeout, EOF, or error."""
        if self._reader is None:
            self._reader = Thread(target=self._read_forever, daemon=True)
            self._reader.start()
 
        # 1. Throw away late answers to an earlier, timed-out question.
        #    Security: a late "y" meant for one tool must never approve another.
        while True:
            try:
                self._answers.get_nowait()
            except QueueEmpty:
                break
            self._pending = False
 
        # 2. Ask for a line, unless a read from a timed-out question is still
        #    waiting. That read simply becomes this question's read.
        if not self._pending:
            self._requests.put(None)
            self._pending = True
 
        # 3. Wait for the answer, redrawing the countdown each second.
        deadline = time() + self.timeout
        while (remaining := deadline - time()) > 0:
            self._tick(remaining)
            try:
                answer = self._answers.get(timeout=min(1.0, remaining))
            except QueueEmpty:
                continue
            self._pending = False
            return answer
        return None  # timed out; the read stays pending for the next question
 
    def _read_forever(self) -> None:
        """Reader thread: one input() per request, never more."""
        while True:
            self._requests.get()
            try:
                line: str | None = self.input_fn("")
            except Exception:  # EOFError (stdin closed, e.g. CI) or anything else
                line = None
            self._answers.put(line)
            
    def _show(self, tool_call: ToolCall, reason: str | None) -> None:
        """Print the question ourselves: a pending read can't print a new prompt."""
        text = self.describe(tool_call, reason)
        # end="": the cursor stays after the question, one line below the countdown.
        if self._live:
            print(f"{text}\n{self._countdown(self.timeout)}\n{self.QUESTION}", end="", flush=True)
        else:
            print(f"{text}\n{self.QUESTION}", end="", flush=True)
 
    def _tick(self, remaining: float) -> None:
        if self._live:
            self._render(self._countdown(remaining), up=1)
 
    def _finish(self, answer: str | None, approved: bool) -> None:
        if not self._live:
            return
        if answer is None:
            self._render("  no answer: denied", up=1)
        else:
            # The human pressed Enter, so the countdown line is now 2 lines up.
            self._render("  approved" if approved else "  denied", up=2)