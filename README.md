# vllmd

Run and orchestrate [vLLM](https://github.com/vllm-project/vllm) model containers — single-node or across a cluster.

## Install

```bash
# Core (single-node Docker management + sessions)
uv pip install -e "."

# With AWS S3 support for sessions and vector store
uv pip install -e ".[aws]"

# With orchestrator + agent daemons (FastAPI / uvicorn / httpx)
uv pip install -e ".[server]"
```

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

## Quick Start

```bash
# Serve a model on port 8000 (foreground — streams vLLM logs until Ctrl+C)
vllmd run --model /path/to/my-model --port 8000

# Run in the background; wait for the API to be ready, then return
vllmd run --model /path/to/my-model --port 8000 -d

# Run in the background and return immediately without waiting
vllmd run --model /path/to/my-model --port 8000 -d --no-wait

# CPU-only (no GPU)
vllmd run --model /path/to/my-model --port 8000 --no-gpu

# Check if the container is up and the API is healthy
vllmd status

# Stream container logs
vllmd logs --follow

# Stop the container
vllmd stop
```

## Multiple Models

Multiple models can run concurrently, each in its own container on a different port. The container name defaults to `vllmd-<model-dir-name>`.

```bash
# Start two models on different ports
vllmd run --model /models/llama3 --port 8001 -d
vllmd run --model /models/mistral --port 8002 -d

# List all running vllmd containers
vllmd ps

# Check health of all containers at once
vllmd status

# Stop a specific container
vllmd stop --name vllmd-llama3

# Stop all vllmd containers
vllmd stop --all
```

When only one container is running, `stop`, `status`, `logs`, and `session create` all auto-resolve to it without needing `--name`.

## How It Works

1. `vllmd run` resolves the model path and pulls `vllm/vllm-openai:latest` if needed
2. A Docker container is started with the model directory mounted read-only at `/model`
3. vLLM serves the model on port 8000 inside the container, mapped to `--port` on the host
4. The served model ID is the directory name of the model path
5. The endpoint exposes a standard OpenAI-compatible API at `http://localhost:<port>/v1`

## Cluster Mode (v2.0+)

Run an **agent** daemon on each GPU node and a single **orchestrator** on the control node. The orchestrator proxies OpenAI-compatible API requests to whichever node is running the requested model.

```bash
# 1. Define nodes and models in vllmd.yaml (see config.example.yaml)

# 2. Start the agent on each GPU node
vllmd agent start --port 7861

# 3. Start the orchestrator on the control node
vllmd orchestrator start --port 7860

# 4. Bring all configured models online
vllmd up

# 5. Point any OpenAI client at the orchestrator
curl http://orchestrator:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b", "messages": [{"role": "user", "content": "Hi"}]}'

# Check node health
vllmd nodes

# Tear everything down
vllmd down
```

Sessions work unchanged — point them at the orchestrator endpoint and use a model name:

```bash
vllmd session create my-session \
  --endpoint http://orchestrator:7860 \
  --model llama3-8b
```

Node pinning: add `X-Vllmd-Node: <name>` to route a request to a specific node.

## Commands

| Command | Description |
|---------|-------------|
| `run` | Start a vLLM container for a model (single-node) |
| `ps` | List all running vllmd containers |
| `stop` | Stop a container (`--all` to stop every managed container) |
| `status` | Show container and API health (all containers if no `--name`) |
| `logs` | Print container logs |
| `up` | Start all (or one) configured models via the orchestrator |
| `down` | Stop all (or one) configured models via the orchestrator |
| `nodes` | List configured nodes and their agent health |
| `agent start` | Start the node agent daemon |
| `agent stop` | Stop the node agent daemon |
| `orchestrator start` | Start the orchestrator service |
| `orchestrator stop` | Stop the orchestrator service |
| `task` | Run an agentic task with tool use against a vLLM endpoint |
| `session create` | Create a persistent chat session |
| `session chat` | Send a one-shot message in a session |
| `session attach` | Open an interactive REPL for a session |
| `session list` | List all sessions |
| `session history` | Print conversation history |
| `session clear` | Clear conversation history |
| `session delete` | Delete a session |
| `db ingest` | Add documents or code to the vector database |
| `db search` | Query the vector database for relevant context |
| `db history` | Store a conversation message |
| `db summarize` | Replace a session's history with an abridged summary |
| `db sync` | Sync the vector DB to/from S3 |
| `db stats` | Show collection sizes |

## Agentic Task Execution (v2.1+)

`vllmd task` runs a prompt through an iterative tool-use loop against any vLLM-compatible endpoint. The model calls tools — bash, file I/O, git — until the task is complete.

```bash
# Run a task against a local model
vllmd task "refactor the auth module to use JWT and commit the result" \
  --endpoint http://localhost:8001 \
  --model llama3 \
  --workdir /path/to/repo

# Use an SSH PEM key for authenticated git push
vllmd task "clone the repo, apply the fix, and push to branch fix/auth" \
  --endpoint http://orchestrator:7860 \
  --model llama3 \
  --workdir /tmp/workspace \
  --pem ~/.ssh/deploy.pem
```

Available tools:

| Tool | Description |
|------|-------------|
| `bash` | Run a shell command; stdout+stderr returned (truncated to 8 KB) |
| `read_file` | Read a file |
| `write_file` | Write a file (creates parent directories) |
| `git_clone` | Clone a repository |
| `git_commit` | Stage all changes and commit |
| `git_push` | Push a branch to a remote |

### `task` options

| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint`, `-e` | (required) | vLLM or orchestrator base URL |
| `--model`, `-m` | (required) | Model ID |
| `--api-key` | — | Bearer token for the endpoint |
| `--workdir`, `-w` | `.` | Working directory for tool execution |
| `--pem` | — | SSH PEM key for git clone/push |
| `--max-turns` | `20` | Maximum conversation turns before stopping |
| `--system` | — | Override the default system prompt |

## Options

### `run`

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | (required) | Path to the model directory (or HuggingFace Hub model ID) |
| `--port`, `-p` | `8000` | Host port for the vLLM API |
| `--name`, `-n` | `vllmd-<model-dir>` | Docker container name |
| `--gpu/--no-gpu` | `--gpu` | Enable/disable GPU passthrough |
| `--dtype` | `auto` | Model dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `--max-model-len` | — | Override max context length |
| `--lora`, `-l` | — | Path to a LoRA adapter directory |
| `--max-lora-rank` | auto | Max LoRA rank (auto-detected from `adapter_config.json` if omitted) |
| `--runtime` | `docker` | Container runtime executable (`docker`, `podman`, …) |
| `--detach`, `-d` | `false` | Start in background |
| `--wait/--no-wait` | `--wait` | Wait for API to be ready (requires `--detach`) |

Extra positional arguments are forwarded verbatim to vLLM.

### `agent start`

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind host |
| `--port`, `-p` | `7861` | Bind port |
| `--runtime` | `docker` | Container runtime (`docker`, `podman`, …) |

### `orchestrator start`

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind host |
| `--port`, `-p` | `7860` | Bind port |

## Sessions

Sessions are persistent, named conversations tied to a running model. Each session maintains sequential conversation history and optionally retrieves semantic context from the vector database.

Sessions are stored as JSON files in `~/.vllmd/sessions/` (override with `--sessions-dir`).

```bash
# Create a session (auto-resolves endpoint if one container is running)
vllmd session create my-session

# Create a session bound to a specific container, with context retrieval
vllmd session create my-session \
  --container vllmd-llama3 \
  --embedding-model llama3 \
  --system-prompt "You are a helpful coding assistant."

# One-shot message
vllmd session chat my-session "Explain the main training loop"

# Interactive REPL (supports /history, /context <query>, /reset, /exit)
vllmd session attach my-session

# View conversation history
vllmd session history my-session --last 10

# List all sessions
vllmd session list

# Clear history (keeps session config)
vllmd session clear my-session

# Delete a session
vllmd session delete my-session
```

### Context retrieval

When a session is created with `--embedding-model`, each message automatically retrieves the most relevant chunks from the session's vector store (documents and code) and injects them as system context before the conversation history. Exchanges are also stored in the ChromaDB history collection for future semantic search.

If the embedding endpoint is unavailable, retrieval is silently skipped and the session continues with history-only context.

## Vector Context Database

vllmd includes a local vector database (backed by [ChromaDB](https://docs.trychroma.com/)) that stores documents, code, and conversation history as embeddings. Embeddings are generated using the same vLLM server the model runs on.

```bash
# Ingest a directory of documents
vllmd db ingest ./docs --type documents --model my-model

# Ingest a codebase
vllmd db ingest ./src --type code --model my-model

# Search for relevant context
vllmd db search "how does auth work" --collection code --model my-model

# Store a conversation message
vllmd db history "Explain the main loop" --role user --session my-session --model my-model

# Abridge old history with a summary
vllmd db summarize --session my-session "Previous conversation covered auth and the main loop." --model my-model

# Push DB to S3
vllmd db sync s3://my-bucket/vectordb --direction push

# Pull DB from S3
vllmd db sync s3://my-bucket/vectordb --direction pull

# Show collection sizes
vllmd db stats
```

The DB directory (`./vectordb` by default, override with `--db-path`) can be mounted as a Docker volume for persistence and shared across machines via S3 sync.

## Using the API

Once running, the endpoint is OpenAI-compatible:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works out of the box with [AgentTester](https://github.com/sroomberg/agenttester):

```yaml
# agent-tester.yaml
agents:
  my-model:
    command: 'agent-tester query http://localhost:8000 my-model {prompt}'
    host: localhost
    commit_style: manual
    timeout: 120
```

## Development

```bash
uv pip install -e ".[dev,aws,server]"
ruff check src/ tests/
ruff format src/ tests/
pytest
```

## Docker

```bash
MODEL_PATH=/path/to/my-model docker compose run --rm vllmd run --model /model --port 8000
```
