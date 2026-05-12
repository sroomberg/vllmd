# Changelog

## v1.0.1 — 2026-05-12
- Deduplicate message deserialization via `_parse_messages()` in `load()` and `list_all()`
- Move embedder factory to `embeddings.make_embedder()`, remove duplicate closures in CLI and chat
- Extract `_post_json()` into `embeddings.py`, use it across chat completion and embedding calls
- Extract `_load_session_or_exit()` to deduplicate session error handling across five commands
- Extract `_ingest_chunks()` to deduplicate `ingest_document` and `ingest_code_file`
- Fix CI publish workflow: move `id-token: write` to workflow level for trusted publishing

## v1.0.0 — 2026-05-12
- Initial stable release
