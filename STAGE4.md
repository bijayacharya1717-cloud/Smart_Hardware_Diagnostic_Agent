# Stage 4: unified diagnostic agent

## Run

1. Activate the project virtual environment and install `requirements-stage4.txt`.
2. Ensure the Qdrant collection `Hardware_Diagnostic` has been populated by the Phase 1–2 indexing scripts.
3. Set `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env`. Optional settings: `GEMINI_MODEL`, `QDRANT_COLLECTION`, `EMBED_MODEL`, and `RERANK_MODEL`.
4. From the project root run:

   ```powershell
   python diagnostic_agent.py "Motor stalls intermittently under load" --failure-time "2026-10-08 13:30:00"
   ```

   Add `--json` to retain the complete tool payload and evidence catalog in machine-readable output.
   Add `--local-only` to skip Gemini and produce a conservative evidence report when Gemini is unavailable. Without this flag, the agent tries the configured Gemini model and supported fallbacks, then produces the same clearly labeled report if all return 503.

The agent uses one FastMCP client configured with three stdio MCP servers: the existing sensor server, a datasheet RAG server, and a schematic RAG server. It discovers each server's tool schemas with `list_tools()`, converts the schemas to Gemini function declarations, and supplies them to Gemini. Gemini chooses which tools to call; the client executes those calls, returns each result to Gemini, and repeats for up to eight tool rounds before requesting a final tool-free synthesis. Tool results receive evidence IDs such as `[E1]`; the final response is expected to cite them. The RAG server runs in two modes so schematic searches filter to `type=diagram_caption`.

The implementation asks anomaly tools about known sensor channels and also tests likely motor channels. Unknown metrics are returned as schema errors. With today's CSV the available channels are temperature, pressure and humidity; therefore a motor-stall report must explicitly say that current and voltage evidence is unavailable. If `--failure-time` is given, the tool checks a ±30-minute target window against a baseline calculated from the entire log, so the fault window does not redefine its own normal level.

## Evaluation

Run `python -m evals.evaluate_fault_scenarios` after configuring DeepEval's judge model credentials. `golden/fault_scenarios.json` includes an injected BME280 over-temperature scenario and a motor-stall scenario that should be handled as insufficient evidence. Contextual recall checks that expected evidence is recovered; faithfulness checks that the diagnosis is supported by retrieved context. Treat automated scores as a regression signal and review high-impact findings manually.

## Safety and limits

This is a portfolio/demo integration over synthetic sensor data. It does not establish causation, control equipment, or replace an engineer. Retrieved passages can be missing or incomplete; the response is required to flag uncertainty, cite evidence, and request relevant measurements. Before using real equipment, validate all sensor calibrations, datasheet revisions, wiring records, time synchronization and evaluation cases.
