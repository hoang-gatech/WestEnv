# SiteDesk

Point-scale site screening for Vietnam. Click a location, get its weather, soil, terrain,
nearby surface water and distance to known dioxin hotspots, from public datasets. The same data
is exposed as [MCP](https://modelcontextprotocol.io) tools, so an LLM can answer questions about
a site and say where each number came from: from Claude Code or LibreChat over MCP, or from the
**Ask** box on the map page.

**Status: slice 2 of 4.** Data tools, MCP server, map UI and a chat assistant work (the chat has
not been run against a live vendor API yet, see below). Not built yet: retrieval over regulations
and past reports with citations, lab-data ingestion and report drafting, and the evaluation
harness (`eval/README.md` has the plan).

```
 browser (MapLibre + Ask box) ──►  FastAPI ──► chat agent ──► Gemini · Claude · OpenAI-compatible
                                      │            │
                                      │            ▼  in-process
 Claude Code · LibreChat ──►  MCP ────┴──►  tools/  ──►  Open-Meteo · SoilGrids · Overpass · Nominatim
                                                 └─ cache (memory + disk) · retries · units checks
```

## Tools

| Tool | Source | Measured latency\* | Notes |
|---|---|---|---|
| `get_weather` | Open-Meteo | ~1 s | model grid cell, not a station |
| `get_soil` | ISRIC SoilGrids 2.0 | 4-16 s, 11.8 s typical | WRB group + probabilities; clay/silt/sand, USDA texture, pH, organic carbon at 0-5, 5-15, 15-30 cm |
| `get_terrain` | Open-Meteo elevation (Copernicus 90 m) | ~1 s | slope under 1.5% reported as flat |
| `get_nearby_water` | OpenStreetMap via Overpass | 2-36 s, intermittent 502s | opt-in: slow and unreliable |
| `get_known_hotspots` | `src/sitedesk/data/hotspots.json` | instant | 3 sites; seed data, see below |
| `reverse_geocode` | Nominatim | ~1 s | 1 request/s limit enforced |
| `get_site_profile` | all of the above | slowest section | partial results; water excluded unless asked |

\*From a single machine on 2026-10-26, a handful of calls each. Treat as orders of magnitude.

Every result carries `location` (the point actually queried), `sources` with retrieval time, and
`caveats`. The server instructions tell the model to quote caveats and never invent a layer
whose tool failed.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate     # Python 3.11+
pip install -e ".[dev]"
export SITEDESK_CONTACT="you@example.com"          # OSM's policies ask for a contact in the User-Agent

python -m sitedesk.api                             # map at http://127.0.0.1:8000
python -m sitedesk.mcp_server                      # MCP over stdio
python -m sitedesk.mcp_server --transport streamable-http --port 8001
```

Use it from Claude Code:

```bash
claude mcp add sitedesk -- "$PWD/.venv/bin/python" -m sitedesk.mcp_server
```

Then ask, for example: *"What are the soil and terrain at 10.978, 106.815, and how close is it
to a known dioxin hotspot?"* A link like `http://127.0.0.1:8000/#10.978,106.815` opens the map on
that point.

Settings (environment): `SITEDESK_CONTACT`, `SITEDESK_CACHE_DIR` (default `~/.cache/sitedesk`;
empty disables the disk cache), `SITEDESK_OVERPASS_URLS` (comma-separated mirrors),
`SITEDESK_CONFIG` (path of the chat config, default `sitedesk.yaml` in the project root).

## Chat: the Ask box

The map page has a third collapsible card, **Ask**. A question goes to `/api/chat`, where a
vendor-neutral agent loop lets the model call the same tools as above, then streams the progress
(`get_soil… 12.1 s ✓`) and the answer back. The point selected on the map is sent along, so "here"
and "this site" work. The loop runs the tools in-process on the MCP server object, so the chat box,
Claude Code and LibreChat always see exactly the same tools, descriptions and error messages.

```bash
pip install -e ".[chat]"                 # one official SDK per vendor (".[dev]" includes it)
cp sitedesk.example.yaml sitedesk.yaml   # git-ignored
cp .env.example .env                     # git-ignored; or export the variables instead
#   ...put a key in .env, for example GEMINI_API_KEY=...
python -m sitedesk.api
```

`sitedesk.yaml` is LibreChat-style: a list of endpoints, each with a provider, a key and models.

```yaml
endpoints:
  - name: gemini
    provider: google                 # google | anthropic | openai (any OpenAI-compatible API)
    apiKey: "${GEMINI_API_KEY}"      # from the environment or the .env next to this file, or the key itself
    models: [gemini-flash-latest]    # the first one is the default
```

| `provider` | SDK | Notes |
|---|---|---|
| `google` | `google-genai` | Gemini function calling; thought signatures are returned untouched, as Gemini 3 requires |
| `anthropic` | `anthropic` | Claude Messages API; adaptive thinking (the model default); `refusal` stop reasons handled; server-side refusal fallbacks on the models that support them (`refusalFallbacks: false` behind Bedrock, Vertex or a gateway) |
| `openai` | `openai` | any OpenAI-compatible Chat Completions API via `baseURL` (OpenAI, OpenRouter, Groq, Mistral, a local server) |

The file is re-read when it changes, so no restart is needed after editing it. Other settings:
`chat.defaultEndpoint`, `chat.maxToolRounds` (default 6), `chat.systemPrompt`, and per endpoint
`maxOutputTokens`. **Keys stay on the server**: they are never in an API response, an error
message or a log line (there are tests for this), and a typo in a key *name* is reported without
echoing the value. Put the keys in `.env` or the environment and keep `sitedesk.yaml` out of git
(it is in `.gitignore`); if you write a literal key into the YAML, that file must never be committed.

What the server enforces: models come from an allow-list in the YAML (the browser cannot pick an
arbitrary one), questions are capped at 4000 characters with at most 20 history entries, at most 3
questions run at once, a question stops after `maxToolRounds` rounds of tool calls, and one tool
result is clipped at 24,000 characters. Each answer shows its tool timings and token count, so cost
per question is a number you can read off the page.

What it does **not** enforce: authentication. The server listens on 127.0.0.1 by default; do not
expose it without adding auth, because anyone who can reach `/api/chat` spends your key's quota.
Also note that Google's pricing page says Gemini free-tier prompts and outputs can be used to improve
its products, so keep the free tier for public data like this, not client data.

## What the upstream services taught me

These shaped the design more than anything else.

- **SoilGrids is slow and its cost grows with properties x depths** (3.6 s for 3 properties at one
  depth, 8-15 s for 5 x 2, ~20 s for 8 x 3 with uncertainty, and the same query varied from 8 to
  15 s). Hence one request for the 5 properties that matter for contaminant behaviour, a 30-day
  disk cache, and the WRB classification fetched concurrently.
- **Overpass answered in ~35 s on several calls, then in 2 s, then with 502s.** Hence: opt-in in
  the UI and in `get_site_profile`, one retry, optional mirrors, and a 7-day cache.
- **SoilGrids returns HTTP 200 with `null` values over water**, and reports integers scaled by a
  per-property factor. The parser checks the reported units against what it labels its fields
  with and fails loudly if they change, instead of mislabelling.
- **The WRB class is often a weak call** (22% for the top class at the Bien Hoa test point), so
  the tool returns the probabilities and adds a low-confidence caveat below 40%.
- **MCP SDK v2 only shows the model the message of a `ToolError`**; any other exception becomes
  a generic "Error executing tool". Upstream failures are mapped at the server boundary so
  Claude can say "SoilGrids timed out" rather than nothing. There is a test for it.
- **MapLibre's `load` event never fired when every initial tile request failed**, which took the
  whole UI down because the click handler lived inside it. The UI now wires itself up directly
  and waits only for `style.load`. Found by running the page in headless Chrome, not by reading
  the code.

## Tests

```bash
pytest                    # 244 tests: logic, recorded real responses, mocked HTTP, real MCP client, chat
pytest -m live -s         # the same tools against the real services, with timings
ruff check . && ruff format --check .
```

The fixtures in `tests/fixtures/` are real responses recorded from the services. Mocked tests
cannot show the services still behave that way, which is what the live tests are for.

The chat is tested at three levels. The agent loop runs with a scripted model against the real
tools. Each vendor adapter runs on that vendor's **real SDK** against a small local fake of its API
(`tests/fake_vendor.py`), which asserts what goes over the wire (paths, headers, bodies, thought
signatures, error mapping) and how the SDK parses a reply. And one test per vendor sends a question
through HTTP, the real agent loop, the real MCP tools, the real SDK and the fake server, so the real
tool schemas and results go through each vendor's serialisation.

The front-end JavaScript has no automated tests: I checked it by hand in headless Chromium with a
scripted fake model (real clicks and keystrokes: ask, tool progress, Stop, errors, the point chip,
an `<img onerror>` in an answer staying inert text, three screen widths, the unconfigured state),
and those scripts are not in the repo.

## Known gaps and things to verify before relying on this

- **`hotspots.json` is seed data.** The three sites are the well-known priority dioxin hotspots at
  former US airbases, but the coordinates are approximate airport reference points from memory
  and I did not check them against a primary source, and there are no citations yet.
- **The USDA texture classifier** is checked at one interior point per class and for full
  coverage of the triangle (no gaps), but not against a reference implementation.
- **The chat has never run against a live vendor API.** There are no keys in my environment, so
  everything above is verified against local fakes and the vendors' documented formats. The first
  real run can still surface a wrong model ID (the ones in `sitedesk.example.yaml` come from the
  docs and a search, and model names change often) or a schema a vendor rejects. A wrong model ID
  or key shows up as a readable error in the chat, not a crash.
- **Licences.** Open-Meteo's free API is for non-commercial use: that needs a paid plan or
  self-hosting before this becomes a commercial product. OSM's public tile servers, Overpass and
  Nominatim are best-effort services with usage policies and are not meant to back a product.
- **Map attribution is hidden while the Site profile panel is open.** OSM, SoilGrids (CC BY 4.0)
  and the terrain tiles are credited in the bottom-right corner of the map, which the open panel
  covers (measured: about 91% of it, on desktop and phone). Collapsing the panel shows it. OSM
  requires the credit to be visible, so this needs fixing before the page is shown to anyone.
- **Not tried yet inside Claude Code or Claude Desktop.** The server is verified with the MCP
  Python client over stdio and streamable HTTP.
- **Administrative names** from OSM may lag Vietnam's 2025 provincial and commune mergers.

## Next

1. Run the chat against a live vendor and read the real numbers off the page: cost and latency
   per question, which tools the model picks, where it invents or hedges.
2. The evaluation harness: the same questions through `chat.agent.run_chat` for each vendor,
   scored on tool choice, grounding and cost (`eval/README.md`).
3. `find_place`: a place-name to coordinates tool, so "soil at Bảo Lộc?" does not rely on the model
   guessing coordinates.
4. Retrieval over public standards and reports (QCVN soil quality, Australian NEPM) with
   citations and a "no source, no answer" rule.
5. Lab CSV in, exceedances against the standard out, a report section drafted into a fixed template.
6. Authentication on `/api/chat`, token streaming, and SoilGrids via cloud-optimised GeoTIFF range
   reads instead of the REST API (which should remove most of the soil latency: expected, not measured).
