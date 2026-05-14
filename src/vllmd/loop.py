"""Agent loop: drives a tool-use conversation with a vLLM-compatible endpoint."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from .tools import TOOL_DEFINITIONS, ToolExecutor

_DEFAULT_SYSTEM = (
    "You are a software engineering assistant with access to tools. "
    "Use them to complete the task. "
    "When you are finished, return a concise summary of what you did."
)


class AgentLoop:
    """Iteratively calls a chat-completion endpoint, executing tool calls until done."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str = "",
        workdir: str = ".",
        pem_path: str | None = None,
        max_turns: int = 20,
        system: str | None = None,
        on_message: Callable[[str, str], None] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.executor = ToolExecutor(workdir=workdir, pem_path=pem_path)
        self.max_turns = max_turns
        self.system = system or _DEFAULT_SYSTEM
        # optional callback(role, content) for streaming output to the terminal
        self.on_message = on_message

    async def run(self, prompt: str) -> str:
        """Run the agent loop starting with *prompt*.

        Returns the final assistant text.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        if self.on_message:
            self.on_message("user", prompt)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=120) as client:
            for _ in range(self.max_turns):
                body = {
                    "model": self.model,
                    "messages": messages,
                    "tools": TOOL_DEFINITIONS,
                    "tool_choice": "auto",
                }
                resp = await client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                msg = choice["message"]
                messages.append(msg)

                finish = choice.get("finish_reason", "")

                if finish != "tool_calls":
                    content = msg.get("content") or ""
                    if self.on_message:
                        self.on_message("assistant", content)
                    return content

                # Execute each tool call and append results
                for tc in msg.get("tool_calls", []):
                    fn = tc["function"]
                    tool_name = fn["name"]
                    args = fn.get("arguments", "{}")
                    tool_result = self.executor.execute(tool_name, args)

                    if self.on_message:
                        preview = tool_result[:200]
                        self.on_message(
                            "tool",
                            f"[{tool_name}] {args} → {preview}",
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result,
                        }
                    )

        return "(max turns reached)"

    def run_sync(self, prompt: str) -> str:
        """Synchronous wrapper around :meth:`run`."""
        import asyncio

        return asyncio.run(self.run(prompt))


def create_loop(
    endpoint: str,
    model: str,
    *,
    api_key: str = "",
    workdir: str = ".",
    pem_path: str | None = None,
    max_turns: int = 20,
    system: str | None = None,
    on_message: Callable[[str, str], None] | None = None,
) -> AgentLoop:
    return AgentLoop(
        endpoint,
        model,
        api_key=api_key,
        workdir=workdir,
        pem_path=pem_path,
        max_turns=max_turns,
        system=system,
        on_message=on_message,
    )
