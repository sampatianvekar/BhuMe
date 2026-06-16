# AI Transcripts

## Web chat links

- **Claude (Anthropic claude.ai)** — used throughout for problem understanding,
  pipeline design, code writing, debugging, and iteration:
  (https://claude.ai/share/b6defa9b-e835-4302-a59f-85cf28ba9a93)

## How AI was used

- Read and summarised the Understand and Task contract pages
- Reasoned through the two error types (placement vs shape/history)
- Designed the 3-stage pipeline architecture
- Wrote and debugged all code in `advanced_solver.py`
- Diagnosed over-flagging issues through data analysis
- Challenged threshold choices and identified generalisation risks

## What the candidate decided

All judgment calls were made by the candidate:
- Which signals to include in confidence and why
- When to stop loosening thresholds (explicitly chose restraint over recovering more plots)
- Which version to freeze and why (documented in the final conversation turn)
- The interpretation of every score result

AI wrote the code; the candidate directed what to build and decided when it was good enough.
