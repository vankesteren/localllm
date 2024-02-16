"""Try out distilgpt2 model on a simple question answering task."""
from transformers import pipeline
from tqdm import tqdm

def fmt(q):
    """Function to format a question into an LLM prompt."""
    return f"You are a participant in a research study. \n\nQuestion: \n{q}\n\nAnswer:"

def generate_answer(question: str, **kwargs) -> str:
    # for args see https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
    resp = pipe(
        fmt(question),
        do_sample=True,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=10.1,
        max_new_tokens=50,
        return_full_text=False,
        early_stopping=True,
        num_beams=4,
        temperature=1.1,
        pad_token_id=pipe.tokenizer.eos_token_id,
        **kwargs,
    )
    return resp[0]["generated_text"]


def generate_answers(question: str, n: int = 1, **kwargs) -> list[str]:
    return [generate_answer(question, **kwargs) for i in tqdm(range(n))]


if __name__ == "__main__":
    print("## Loading model...")
    pipe = pipeline("text-generation", model="./distilgpt2", device="cuda")

    item = "How do you deal with anxiety before having to give a presentation?"
    print(f"## Generating answers to the following question:\n   {item}")
    responses = generate_answers(item, 10)

    print("## Here are the responses:\n")
    for r in responses:
        print(f"- {r}")
