from .policies import Policy
from .errors import ToolError, MaxStepsReachedError, PolicyAbortError
from .executor import Executor
from .state import ConversationState
from .approval import ToolApprover
from .messages import Message, ToolCall, ToolResult
from tools.registry import ToolRegistry
from models.base import ModelClient
import json

DenialCounts = dict[tuple[str, str], int] # (tool name, canonical args) -> times denied

class Runtime:
    def __init__(self, model: ModelClient, tool_registry: ToolRegistry, executor: Executor, policy: Policy, approver: ToolApprover, max_steps: int, max_denials: int):
        
        self.model = model
        self.tool_registry = tool_registry
        self.executor = executor
        self.policy = policy
        self.approver = approver
        self.max_steps = max_steps
        self.max_denials = max_denials
        
    def run(self, state: ConversationState, prompt: str) -> str:
        """Run the conversation with the given prompt."""
        
        # Add the user prompt to the conversation history
        state.add(Message.user(prompt))
        
        # Get the tools from the registry
        specs = self.tool_registry.specs()
        
        # Trace the tool denials (ToolName -> DenialCount)
        denials: DenialCounts = {}
        
        # Iterate over the steps
        for step in range(self.max_steps):
            
            response = self.model.generate(state.messages, specs)
            state.add(Message.assistant(response))
            
            if not response.tool_calls:
                return response.response or ""
            
            # Process the tool calls
            for i, call in enumerate(response.tool_calls):
                try:
                    result = self.handle(call, denials)
                except BaseException:
                    # Every tool call needs a result, or the provider rejects the history on the next run.
                    for left in response.tool_calls[i:]:
                        state.add(Message.tool(ToolResult(
                            tool_name=left.tool_name, tool_id=left.tool_id,
                            status="error", tool_result="Not run: the run was stopped.")))
                    raise
                
                # Add the tool response to the conversation history
                state.add(Message.tool(result= result))
            
        # Raise MaxStepsReachedError if the conversation exceeds the max steps
        raise MaxStepsReachedError(f"Stopped after {self.max_steps} steps without a final answer")
        
    def handle(self, call: ToolCall, denials: DenialCounts) -> ToolResult:
        """Handle a tool call."""
        
        # Check if the tool exists
        if call.tool_name not in self.tool_registry:          
            # list all tools available
            available = "; ".join(sorted(self.tool_registry))
            return ToolResult(tool_name=call.tool_name, tool_id=call.tool_id, status="error", tool_result=f"Tool {call.tool_name} not found, available tools: {available}")
        
        # Check for argument errors
        if call.arg_error:
            return ToolResult(tool_name=call.tool_name, tool_id=call.tool_id, status="error", tool_result=f"{call.tool_name}: {call.arg_error}")
        
        # Check policy
        decision = self.policy.evaluate(call)
        
        if decision.action == "ask":
            if not self._ask(call, decision.reason):
                return self._deny(call, denials)
            
        elif decision.action != "allow":
            return self._deny(call, denials)
        
        try:
            return self.executor.run(call)
        except ToolError as e:
            return ToolResult(tool_name=call.tool_name, tool_id=call.tool_id, status="error", tool_result=str(e))
        
    
    def _ask(self, call: ToolCall, reason: str | None) -> bool:
        try:
            return self.approver.approve(call, reason)
        except Exception:
            return False
        
    def _deny(self, call: ToolCall, denials: DenialCounts) -> ToolResult:
        """Both kinds of "no" (a rule, or a human/timeout) land here and look identical."""
        key = (call.tool_name, json.dumps(call.tool_args, sort_keys=True, default=str))
        denials[key] = denials.get(key, 0) + 1
        
        if denials[key] >= self.max_denials:
            raise PolicyAbortError(f"The model kept retrying a denied call to '{call.tool_name}'")
        return ToolResult(tool_name=call.tool_name, tool_id=call.tool_id, status="denied", tool_result="Denied. Do not retry.")