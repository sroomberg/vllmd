"""OpenAI-format tool definitions for the agent loop."""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command and return its stdout + stderr. "
                "Output is truncated to 8 KB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Working directory (defaults to the task workdir)."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the text content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file, creating parent dirs as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_clone",
            "description": "Clone a git repository into a local directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Repository URL (https:// or git@).",
                    },
                    "dest": {
                        "type": "string",
                        "description": "Destination directory (defaults to repo name).",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch or tag to check out.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes (git add -A) and create a commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Repository directory (defaults to task workdir)."
                        ),
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Push a local branch to a remote.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name to push.",
                    },
                    "remote": {
                        "type": "string",
                        "description": "Remote name (default: origin).",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Repository directory (defaults to task workdir)."
                        ),
                    },
                },
                "required": ["branch"],
            },
        },
    },
]
