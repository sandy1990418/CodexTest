# CodexTest

This repository demonstrates a minimal example of using the `dspy` Python
library. The `dspy_example.py` script defines a `Signature` and wraps it
inside a `Module` to run a summarization task.  It also shows how to set up
an optimizer and call `dspy.compile` to tune the module.

## Running the example

1. Install the library:

   ```bash
   pip install dspy
   ```

   You'll also need an OpenAI API key for the LLM.

2. Run the script:

    ```bash
    python dspy_example.py
    ```

   The example imports `OpenAI` from `dspy.llm`, so make sure you
   have a recent version of DSPy installed.

The script defines a simple dataset, builds a summarizer `Module`, and then
uses `dspy.compile` with a basic optimizer to tune the module on that dataset
before running it.
