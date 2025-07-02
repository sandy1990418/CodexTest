import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO
import os

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

class SentimentClassifier(dspy.Signature):
    """Define the input and output fields for SentimentClassifier."""
    text: str = dspy.InputField(desc="text")
    sentiment: str = dspy.OutputField(desc="sentiment")

class Summarizer(dspy.Module):
    """Simple module that generates a SentimentClassifier using an LLM."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SentimentClassifier)

    def forward(self, text: str) -> str:
        """Return a summary for the given document."""
        result = self.predict(text=text)
        return result.sentiment

def accuracy_metric(example, pred):
    return int(example.sentiment.strip().lower() == pred.strip().lower())


if __name__ == "__main__":
    # Example training data for optimization/compilation
    trainset = [
        {"text": "I love this product!", "sentiment": "positive"},
        {"text": "It was a horrible experience.", "sentiment": "negative"},
        {"text": "The food was amazing!", "sentiment": "positive"},
        {"text": "Worst customer service ever.", "sentiment": "negative"},
    ]

    devset = [
        {"text": "This app is so useful!", "sentiment": "positive"},
        {"text": "I'm really disappointed with the quality.", "sentiment": "negative"},
    ]

    dataset = []
    testset = []

    for context in trainset:
        text = context.get("text")
        sentiment = context.get("sentiment")
        dataset.append(dspy.Example(text=text,sentiment=sentiment).with_inputs("text"))

    for context in devset:
        testset.append(dspy.Example(text=context.get("text"), sentiment=context.get("sentiment")).with_inputs("text"))
    
    # Initialize an LLM (replace with your credentials)
    llm = dspy.LM(model="gpt-4o-mini")
    dspy.configure(lm=llm)

    # Build the module
    summarizer = Summarizer()
    response = summarizer(text="This app is so useful!")
    print(response)

    # Set up an optimizer and compile the module on the dataset
    optimizer  = COPRO(metric=accuracy_metric, max_iters=3, verbose=True, depth=10)
    eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)
    copro_model = optimizer.compile(summarizer, trainset=dataset,eval_kwargs=eval_kwargs)

    Evaluate(devset=testset, metric=accuracy_metric, display_progress=True)(copro_model)
