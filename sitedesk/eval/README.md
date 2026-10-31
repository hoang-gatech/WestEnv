# Evaluation (not built yet)

This folder will hold the harness that scores the assistant, not just the tools. The tools are
already covered by `tests/`; what needs measuring is the model's behaviour on top of them.

Planned, in order:

1. **Tool-choice cases** (`cases/*.jsonl`): `{"question", "lat", "lon", "expect_tools": [...],
   "forbid_tools": [...]}`. Did it call `get_soil` for a soil question, and skip `get_nearby_water`
   (up to a minute) when water was not asked about?
2. **Grounding cases**: every number in the answer must trace to a tool result; a question whose
   tool failed must be answered with "unavailable", never with an invented value.
3. **Citation cases** (once the document index exists): answer must cite a retrieved passage, and
   refuse when nothing relevant was retrieved.
4. **Cost and latency per question**, recorded with every run so regressions in either show up.

Results go in `runs/` as JSON, one file per run, with the model, prompt version and date.
