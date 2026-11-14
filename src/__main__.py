from llm_sdk import Small_LLM_Model
from src.parser import get_functions, get_prompts
import numpy as np
import json



# prompt = f"You are to output strict JSON format and choose one of these functions as the answer:\n" \
# f"{get_functions()}\n\n"
names = "functions: ft_add(), ft_multiply, ft_divide, ft_substract"


query = "Question:\n" \
"one plus two equals "

all_prompt = "" + query

model = Small_LLM_Model()

def get_vocab() -> dict:
    vocab = model.get_path_to_vocab_file()
    with open(vocab, "r", encoding="utf-8") as f:
        return json.load(f)

def unmask_logits(vocab):
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    capitalized = alphabets.upper()
    nums = "0123456789"
    special = " ,:\\.\n*-/?!%&-_+}{()"
    thesum = alphabets


    indices = []
    for key, value in vocab.items():
        key = key.replace("Ġ", " ").replace(" ", " ")
        if all(c in thesum for c in key):
            indices.append(value)
    return indices


vocab = get_vocab()
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=4)

loop = 1
while loop:

    prompt_encoded = model.encode(all_prompt)[0]


    logits = model.get_logits_from_input_ids(prompt_encoded.tolist())

    # this is the step where we can mask
    masked = np.array(logits)
    masked[:] = -np.inf

    # this is the step where we can unmask
    indices = unmask_logits(vocab)
    not_masked = np.array(logits)
    masked[indices] = not_masked[indices]
    # for indice in indices:
    #     masked[indice] = logits[indice]
    


    highlogit = np.argmax(masked)

    char = model.decode(highlogit)

    all_prompt += char
    print(char, end="", flush=True)
    loop -= 1