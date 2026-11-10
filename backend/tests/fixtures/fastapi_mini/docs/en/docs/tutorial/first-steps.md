# First Steps { #first-steps }

The simplest FastAPI file could look like this:

{* ../../docs_src/first_steps/tutorial001.py hl[3] *}

Copy that to a file `main.py`.

## Run it { #run-it }

Run the live server with the `fastapi dev` command.

```Python
{!../../docs_src/first_steps/tutorial001.py!}
```

### Check it { #check-it }

Open your browser at `http://127.0.0.1:8000`. Only the route lines:

{* ../../docs_src/first_steps/tutorial001.py ln[6:9] *}

## Recap

You import `FastAPI`, create an `app` instance and write a path operation decorator.
