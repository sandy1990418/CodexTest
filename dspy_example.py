import dspy

class SummarizeSignature(dspy.Signature):
    """Define the input and output fields for summarization."""
    doc: str = dspy.InputField(desc="Document to summarize")
    summary: str = dspy.OutputField(desc="Concise summary")

class Summarizer(dspy.Module):
    """Simple module that generates a summary using an LLM."""
    def __init__(self, llm=None):
        super().__init__()
        self.predict = dspy.Predict(SummarizeSignature, llm=llm)

    def forward(self, doc: str) -> str:
        """Return a summary for the given document."""
        result = self.predict(doc=doc)
        return result.summary

if __name__ == "__main__":
    # Example training data for optimization/compilation
    trainset = [
        {"doc": "DSPy helps structure prompts for LLMs.",
         "summary": "DSPy structures prompts."},
        {"doc": "Modules in DSPy make pipelines reusable.",
         "summary": "Modules enable reusable pipelines."}
    ]

    # Initialize an LLM (replace with your credentials)
    llm = dspy.OpenAI(api_key="YOUR_API_KEY")

    # Build the module
    summarizer = Summarizer(llm=llm)

    # Set up an optimizer and compile the module on the dataset
    optimizer = dspy.optim.SimpleOptimizer(max_iters=2)
    compiled = dspy.compile(
        module=summarizer,
        trainset=trainset,
        evalset=trainset,
        optimizer=optimizer,
        llm=llm,
    )

    # Run the compiled module
    text = "DSPy helps structure prompts for large language models."
    print(compiled(doc=text))
