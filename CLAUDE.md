# kernel

## Writing the paper

Before writing or editing `ISFPGA_Submission/main.tex`, read
`ISFPGA_Submission/persona.md` and write in that voice. This is not optional
and not a style preference applied afterwards — it is how this project reports
results, and prose that does not match it gets rewritten.

## Environment

The system `python3` is 3.6 and cannot parse this source. Use
`/home/digital3/TurboQuant-Reproduction/.venv/bin/python` (3.14, with numpy,
torch, transformers, datasets).

Tests: `python -m kernel.tests.run [pattern]`, run from `/home/digital3`.
There is no pytest; a test is a module-level `check_*` function that raises.
