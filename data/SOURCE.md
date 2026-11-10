# Corpus source and licence

The corpus is the English documentation of [FastAPI](https://github.com/fastapi/fastapi),
pinned to release tag `0.162.4` (commit `e10178c18d8dc2dcc21a70e14152253abc4a4e2f`).

## Licence

FastAPI, including its documentation and code samples, is released under the MIT licence:

> The MIT License (MIT)
>
> Copyright (c) 2018 Sebastián Ramírez

The download script (`rag-arena corpus download`) reads the `LICENSE` file at the pinned ref
and stops if it is not the MIT licence. MIT allows reuse, modification and redistribution as
long as the copyright notice and permission notice are kept. This repository does not commit
the documentation itself: `data/raw/` is gitignored and re-created by the script. The eval set
in `data/eval/qa.jsonl` quotes short evidence passages from the docs; those quotes remain
Copyright (c) 2018 Sebastián Ramírez under the MIT licence, whose full text is in the
downloaded `data/raw/fastapi-0.162.4/LICENSE` and at
<https://github.com/fastapi/fastapi/blob/0.162.4/LICENSE>.

## What is downloaded

`https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/0.162.4`, from which only these
paths are extracted:

- `docs/en/docs/**`: the English docs pages
- `docs_src/**`: code samples the pages include
- `fastapi/**`: one page includes lines from the library source
- `LICENSE`

## Processing

- Include directives (`{* ../../docs_src/x.py ln[3:9] hl[4] *}`, `{!> ../../docs_src/x.py!}`)
  are replaced with the referenced code, honouring `ln[...]` line selections.
- Explicit heading anchors (`{ #anchor }`) are removed from the text and kept as metadata.
- Excluded pages: `release-notes.md`, `reference/**`, `fastapi-people.md`,
  `external-links.md`, `management.md`, `newsletter.md`, `_llm-test.md`, `agent-banner.md`,
  `translation-banner.md`. They are generated, stubs for docstring-rendered API reference, or
  not documentation.

The result is 126 documents and 975,860 characters. `data/manifest.json` lists every processed
document with its sha256; `rag-arena corpus verify` checks a fresh download against it.
