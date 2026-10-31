# SiteDesk

Point-scale site screening for Vietnam. Click a location, get its weather, soil, terrain,
nearby surface water and distance to known dioxin hotspots, from public datasets. The same data
is exposed as [MCP](https://modelcontextprotocol.io) tools so Claude can answer questions about
a site and say where each number came from.

**Status: slice 1 of 4.** Data tools, MCP server and map UI work. Not built yet: the Claude agent
loop, retrieval over regulations and past reports with citations, lab-data ingestion and report
drafting, and the evaluation harness (`eval/README.md` has the plan).

```
 browser (MapLibre)  ──►  FastAPI  ──┐
                                     ├──►  tools/  ──►  Open-Meteo · SoilGrids · Overpass · Nominatim
 Claude Code / Desktop ──►  MCP  ────┘          │
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
empty disables the disk cache), `SITEDESK_OVERPASS_URLS` (comma-separated mirrors).

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
pytest                    # 133 tests: logic, recorded real responses, mocked HTTP, real MCP client
pytest -m live -s         # the same tools against the real services, with timings
ruff check . && ruff format --check .
```

The fixtures in `tests/fixtures/` are real responses recorded from the services. Mocked tests
cannot show the services still behave that way, which is what the live tests are for. The
front-end JavaScript has no automated tests: I checked it by hand in headless Chrome (real click,
every overlay, a hash link, phone width), and those scripts are not in the repo.

## Known gaps and things to verify before relying on this

- **`hotspots.json` is seed data.** The three sites are the well-known priority dioxin hotspots at
  former US airbases, but the coordinates are approximate airport reference points from memory
  and I did not check them against a primary source, and there are no citations yet.
- **The USDA texture classifier** is checked at one interior point per class and for full
  coverage of the triangle (no gaps), but not against a reference implementation.
- **Licences.** Open-Meteo's free API is for non-commercial use: that needs a paid plan or
  self-hosting before this becomes a commercial product. OSM's public tile servers, Overpass and
  Nominatim are best-effort services with usage policies and are not meant to back a product.
  Attribution is shown in the map for OSM, SoilGrids (CC BY 4.0) and the terrain tiles.
- **Not tried yet inside Claude Code or Claude Desktop.** The server is verified with the MCP
  Python client over stdio and streamable HTTP.
- **Administrative names** from OSM may lag Vietnam's 2025 provincial and commune mergers.

## Next

1. Agent loop: Claude with these tools over MCP, answers grounded and cited, with the eval
   harness to measure it.
2. Retrieval over public standards and reports (QCVN soil quality, Australian NEPM) with
   citations and a "no source, no answer" rule.
3. Lab CSV in, exceedances against the standard out, a report section drafted into a fixed template.
4. SoilGrids via cloud-optimised GeoTIFF range reads instead of the REST API, which should remove
   most of the soil latency (expected, not measured).
