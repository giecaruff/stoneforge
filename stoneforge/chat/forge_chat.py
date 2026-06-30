"""ForgeChat: a lightweight natural-language interface for scientific tools.

This module defines ForgeChat, a class that keeps conversation history, tool
results, and a session state. It supports a pluggable local LLM callable
and registering/calling local tools. The LLM is expected to be a callable
that accepts a prompt (str) and returns a string response. Tool calls are
performed via registered Python callables.

Usage example:
    from stoneforge.chat.forge_chat import ForgeChat

    def echo_tool(text):
        return f"ECHO: {text}"

    def dummy_llm(prompt: str) -> str:
        return "{"\"assistant\": \"I can call tools with JSON: {\\\"tool_call\\\": {\\\"name\\\": \\\"echo\\\", \\\"args\\\": {\\\"text\\\": \\\"hello\\\"}}}\"}"  # pragma: no cover

    fc = ForgeChat(llm_callable=dummy_llm)
    fc.register_tool("echo", echo_tool, "Echoes text")
    resp = fc.respond("Say hello and use the echo tool")
    print(resp)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional,Annotated, get_args, get_origin

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # 'user' | 'assistant' | 'system'
    text: str


@dataclass
class ToolSpec:
    func: Callable[..., Any]
    description: str = ""


import inspect
from functools import wraps
from typing import get_type_hints


def _parse_param_descriptions(doc: str) -> Dict[str, str]:
    """Very small parser extracting simple ``:param name: desc`` and NumPy-style
    "Parameters" sections. Falls back to empty descriptions when parsing fails.
    """
    if not doc:
        return {}
    lines = doc.splitlines()
    res: Dict[str, str] = {}
    # Sphinx-style: :param name: description
    for line in lines:
        s = line.strip()
        if s.startswith(":param"):
            # format ":param name: desc"
            parts = s.split(None, 2)
            if len(parts) >= 3:
                name = parts[1].rstrip(":")
                desc = parts[2].lstrip(": ")
                res[name] = desc
    # NumPy-style: find a 'Parameters' section and parse entries like 'name : type'
    try:
        idx = next(i for i, l in enumerate(lines) if l.strip().startswith("Parameters"))
    except StopIteration:
        return res
    i = idx + 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # entry like: param_name : type
        if ":" in line:
            parts = line.split(":", 1)
            name = parts[0].strip()
            # collect indented description lines
            desc_lines: List[str] = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
                desc_lines.append(lines[j].strip())
                j += 1
            res.setdefault(name, " ".join([dl for dl in desc_lines if dl]))
            i = j
            continue
        i += 1
    return res


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark callables as tools and attach extracted metadata.

    The decorator inspects the function signature, type hints, and docstring to
    produce a _tool_metadata attribute on the wrapped function with the keys:
      - name: tool name
      - description: short description
      - doc: full docstring
      - params: list of {name, type, description}
      - return_annotation: return type name or raw annotation
    """
    def decorator(func: Callable[..., Any]):
        meta_name = name or func.__name__
        doc = func.__doc__ or ""
        sig = inspect.signature(func)
        hints = get_type_hints(
            func,
            include_extras=True
        )
        param_descs = _parse_param_descriptions(doc)
        params: List[Dict[str, Any]] = []
        for pname, param in sig.parameters.items():
            ann = None
            if pname in hints:
                ann = hints[pname]
            elif param.annotation is not inspect._empty:
                ann = param.annotation
            ann_name = None
            ann_description = param_descs.get(pname, "")

            if ann is not None:

                # Handle Annotated[T, description]
                if get_origin(ann) is Annotated:

                    annotated_args = get_args(ann)

                    if annotated_args:
                        base_type = annotated_args[0]

                        try:
                            ann_name = base_type.__name__
                        except AttributeError:
                            ann_name = str(base_type)

                        # First metadata entry is usually the description
                        if len(annotated_args) > 1:
                            ann_description = str(annotated_args[1])

                else:

                    try:
                        ann_name = ann.__name__
                    except AttributeError:
                        ann_name = str(ann)

            params.append(
                {
                    "name": pname,
                    "type": ann_name,
                    "description": ann_description,
                }
            )
            
        return_ann = hints.get("return", None)
        try:
            return_ann_name = return_ann.__name__ if return_ann is not None else None
        except Exception:
            return_ann_name = str(return_ann)
        metadata = {
            "name": meta_name,
            "description": description or (doc.splitlines()[0] if doc else ""),
            "doc": doc,
            "params": params,
            "return_annotation": return_ann_name,
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._tool_metadata = metadata
        return wrapper
    return decorator


def get_tool_metadata(func: Callable[..., Any]) -> Optional[Dict[str, Any]]:
    return getattr(func, "_tool_metadata", None)


class ForgeChat:
    """Chat interface that maintains history, tool outputs, and session state.

    Arguments:
        llm_callable: callable(prompt: str) -> str
            A local LLM inference function. If None, run_llm raises.
        system_prompt: optional system prompt prepended to all prompts.
    """

    def __init__(self, llm_callable: Optional[Callable[[str], str]] = None, system_prompt: str = ""):
        self.llm_callable = llm_callable
        self.system_prompt = system_prompt
        self.history: List[Message] = []
        self.tools: Dict[str, ToolSpec] = {}
        self.tool_results: Dict[str, List[Any]] = {}
        self.session_state: Dict[str, Any] = {}

    # Conversation management
    def add_user_message(self, text: str) -> None:
        self.history.append(Message(role="user", text=text))

    def add_assistant_message(self, text: str) -> None:
        self.history.append(Message(role="assistant", text=text))

    def add_system_message(self, text: str) -> None:
        self.history.append(Message(role="system", text=text))

    def get_history(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "text": m.text} for m in self.history]

    # Tool registration & calling
    def register_tool(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        if name in self.tools:
            logger.warning("Overwriting tool %s", name)
        self.tools[name] = ToolSpec(func=func, description=description)
        self.tool_results.setdefault(name, [])

    def call_tool(self, name: str, *args, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered")
        logger.debug("Calling tool %s with args=%s kwargs=%s", name, args, kwargs)
        result = self.tools[name].func(*args, **kwargs)
        # store result in history for transparency
        self.tool_results.setdefault(name, []).append(result)
        # add assistant note about tool result
        short = f"[tool:{name}] {str(result)}"
        self.add_assistant_message(short)
        return result

    # LLM inference
    def run_llm(self, prompt: str, **kwargs) -> str:
        if not self.llm_callable:
            raise RuntimeError("No LLM callable provided")
        return self.llm_callable(prompt, **kwargs)

    def build_prompt(self) -> str:
        parts: List[str] = []
        if self.system_prompt:
            parts.append(f"System:\n{self.system_prompt}\n---\n")
        for m in self.history:
            parts.append(f"{m.role.upper()}: {m.text}\n")
        # include brief tool inventory
        if self.tools:
            inventory = {k: v.description for k, v in self.tools.items()}
            parts.append("TOOLS: " + json.dumps(inventory) + "\n")
        return "\n".join(parts)

    # Simple parser expecting assistant to emit JSON with an optional tool_call field:
    # Example assistant response:
    # {"assistant": "...", "tool_call": {"name":"echo","args": {"text":"hi"}}}
    @staticmethod
    def _try_parse_tool_call(llm_text: str) -> Optional[Dict[str, Any]]:
        # tries to extract JSON object from LLM text
        try:
            # find first { and last }
            start = llm_text.find("{")
            end = llm_text.rfind("}")
            if start == -1 or end == -1:
                return None
            j = json.loads(llm_text[start : end + 1])
            if isinstance(j, dict) and "tool_call" in j:
                return j["tool_call"]
        except Exception:
            logger.debug("Failed to parse tool call from LLM text", exc_info=True)
        return None

    def respond(self, user_input: str, call_tools: bool = True, **llm_kwargs) -> str:
        """Add user input, run LLM, optionally dispatch tool calls, and return assistant text.

        If the LLM returns JSON containing a 'tool_call' object, and call_tools=True,
        the specified tool will be invoked with provided args dict.
        """
        self.add_user_message(user_input)
        prompt = self.build_prompt()
        prompt += f"USER: {user_input}\nASSISTANT:"  # hint for the model
        llm_out = self.run_llm(prompt, **llm_kwargs)
        # store raw assistant message
        self.add_assistant_message(llm_out)
        # attempt to parse a tool call
        tool_call = self._try_parse_tool_call(llm_out)
        if tool_call and call_tools:
            name = tool_call.get("name")
            args = tool_call.get("args", {})
            if not name:
                logger.warning("Parsed tool_call without a name: %s", tool_call)
                return llm_out
            try:
                result = self.call_tool(name, **args)
                # return combined response with tool output
                combined = json.dumps({"assistant": llm_out, "tool_result": result})
                # add combined note to history
                self.add_assistant_message(combined)
                return combined
            except Exception as e:
                err = f"Tool call failed: {e}"
                self.add_assistant_message(err)
                return json.dumps({"assistant": llm_out, "error": err})
        return llm_out

    def register_decorated_tool(self, func):
        meta = get_tool_metadata(func)

        if meta is None:
            raise ValueError(
                f"{func.__name__} is not decorated with @tool"
            )

        self.register_tool(
            meta["name"],
            func,
            meta["description"]
        )

# Minimal demo LLM (for local testing). Real integration should pass a real inference function.
def simple_llm(prompt: str) -> str:
    """Very small heuristic LLM stub used for local testing only.

    Looks for the token "use echo" and emits a tool_call JSON to call a registered
    'echo' tool with a text argument. Otherwise echoes back.
    """
    if "use echo" in prompt.lower():
        payload = {"assistant": "I'll call the echo tool.", "tool_call": {"name": "echo", "args": {"text": "hello from simple_llm"}}}
        return json.dumps(payload)
    return json.dumps({"assistant": "No tools needed. Echoing back."})


if __name__ == "__main__":
    # Demo quick run
    logging.basicConfig(level=logging.DEBUG)
    fc = ForgeChat(llm_callable=simple_llm, system_prompt="You are ForgeChat: assist with tools.")

    def echo(text: str) -> str:
        return f"ECHOED: {text}"

    fc.register_tool("echo", echo, "Echoes input text")
    print(fc.respond("Please use echo to greet me."))
