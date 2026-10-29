from typing import Iterable, Any
from pathlib import Path

from models.base import ModelClient
from .messages import Message
from .runtime import Runtime
from .executor import Executor
from .policies import Policy, Action
from .approval import ToolApprover, CLIApprover
from tools.tool import Tool
from tools.registry import ToolRegistry
from .state import ConversationState
from .memory import MemoryStore, make_memory_tools
from tools.default_tools import default_tools, default_policy, SearchFn
from ._prompts import build_system_prompt


Rules = dict[str, Action]

MEMORY_TOOLS = frozenset({"save_to_memory", "delete_from_memory"})


class Agent:
    def __init__(self, model: str | ModelClient, tools: Iterable[Tool] = (), policy: Policy | Rules | None = None, *, deflt_tools: bool = True, root: str | Path = ".",
                 search_fn: SearchFn | None = None, memory: MemoryStore | bool = True, approver: ToolApprover | None = None, system_prompt: str | None = None, instructions: str | None = None,
                 max_steps: int = 20, max_denials: int = 3,tool_timeout: float = 30.0, **model_kwargs: Any) -> None:
        
        # 1. Set Model client
        if isinstance(model, str):
            from models.modelClient import LiteLLMClient
            self.model: ModelClient = LiteLLMClient(model, **model_kwargs)
        else:
            self.model = model
        
        # 2. Memory: True -> the default store, False -> no memory at all.
        if isinstance(memory, MemoryStore):
            self.memory: MemoryStore | None = memory
        else:
            self.memory = MemoryStore() if memory else None
        
        # 3. Tools: defaults, then memory's, then the caller's
        self.tools = ToolRegistry(default_tools(root, search_fn) if deflt_tools else ())

            # Register memory tools
        if self.memory is not None:
            for t in make_memory_tools(self.memory):
                self.tools.register(t)
                
            # Register caller's tools (replace if already registered)
        for t in tools:
            self.tools.register(t, replace= True)
            
        # 4. Policy: Policy() replaces the base_policy; a Rule overrides.
        if isinstance(policy, Policy):
            self.policy = policy
        else:
            base_policy = default_policy().override({name: "ask" for name in MEMORY_TOOLS})
            self.policy = base_policy.override(policy) if policy else base_policy
            
        # 5. Runtime
        self._approver = approver if approver is not None else CLIApprover()
        self._executor = Executor(self.tools, timeout= tool_timeout)
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = build_system_prompt(root=Path(root), memory=self.memory is not None, instructions=instructions)
        
        self.runtime = Runtime(self.model, self.tools, self._executor, self.policy, self._approver, max_steps, max_denials)
        
        # 6. Conversation state
        self.reset()
        
    def reset(self) -> None:
        """Start a new conversation, with saved memory read fresh into the system prompt."""
        prompt = self.system_prompt
        if self.memory is not None and (saved := self.memory.as_prompt()):
            prompt = f"{prompt}\n\n{saved}" if prompt else saved
        self.state = ConversationState([Message.system(prompt)] if prompt else [])
        
    def run(self, prompt: str) -> str:
        """Run the conversation with the given prompt."""
        return self.runtime.run(self.state, prompt)
        
