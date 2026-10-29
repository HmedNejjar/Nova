from dataclasses import dataclass
from typing import Literal
from .messages import ToolCall

Action = Literal["allow", "ask", "deny"]
 
# Strictest wins: deny > ask > allow.
_RANK: dict[Action, int] = {"allow": 0, "ask": 1, "deny": 2}


@dataclass(frozen=True)
class Decision:
    """The policy's verdict on one ToolCall."""
    action: Action
    reason: str | None = None
    
@dataclass(frozen=True)
class Policy:
    
    allow: frozenset[str] = frozenset()   
    deny: frozenset[str] = frozenset()
    ask: frozenset[str] = frozenset()
    
    default: Action = "ask"
    
    def evaluate(self, tool_call: ToolCall) -> Decision:
        
        # 1. Collect an opinion from every list that mentions this tool.
        opinions: list[Decision] = []
        
        # Check if the tool name is in any of the policy set
        if tool_call.tool_name in self.allow:
            opinions.append(Decision(action="allow", reason="tool in allow list"))
        if tool_call.tool_name in self.deny:
            opinions.append(Decision(action="deny", reason="tool in deny list"))
        if tool_call.tool_name in self.ask:
            opinions.append(Decision(action="ask", reason="tool in ask list"))
        
        # 2. If no opinion, return default.
        if not opinions:
            return Decision(action=self.default, reason=f"no rule for this tool; default is {self.default}")
        
        # 3. Return the opinion with highest rank.
        return max(opinions, key=lambda d: _RANK[d.action])
            
    def override(self, new_rules: dict[str, Action]) -> "Policy":
        """Override the policy with new rules for each tool in new_rules
            where `new_rules` is a dict of tool_name -> action."""
        names = frozenset(new_rules)
        by_action = {a: frozenset(n for n, act in new_rules.items() if act == a) for a in _RANK}
        return Policy(allow=(self.allow - names) | by_action["allow"],
                      ask=(self.ask - names) | by_action["ask"],
                      deny=(self.deny - names) | by_action["deny"],
                      default=self.default)
        
        