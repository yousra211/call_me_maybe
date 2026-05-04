import json
from llm_sdk import llm_sdk
import numpy as np

with open('function_calling_tests.json', 'r') as f:
    data = json.load(f)

my_model = llm_sdk.Small_LLM_Model()

with open('functions_definition.json', 'r') as f:
    functions_definition = json.load(f)

functions_text = ""
for fn in functions_definition:
    functions_text += f"- {fn['name']}: {fn['description']}\n"
    functions_text += f"  parameters: {json.dumps(fn['parameters'])}\n"

prompt = f"""<|im_start|>system
You are a function calling assistant. Your only job is to return a JSON object representing which function to call and with what arguments. Never solve the problem yourself.<|im_end|>
<|im_start|>user
Available functions:
{functions_text}

Example output format:
{{"name": "fn_add_numbers", "parameters": {{"a": 1, "b": 2}}}}

Now return the correct JSON for this request: {data[0]['prompt']}
<|im_end|>
<|im_start|>assistant
{{"""
encoded = my_model.encode(prompt)
mylist = encoded[0].tolist()
open_brackets = 1
eos_token_id = my_model.encode("<|endoftext|>").tolist()[0]

generated_ids = []

vocab_path = my_model.get_path_to_vocab_file()

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# invert it to get id → token
id_to_token = {v: k for k, v in vocab.items()}
state = "START"

def get_valid_tokens(state: str, vocab: dict) -> list:
    valid_ids = []
    if state == "START":
        valid_ids.append(vocab['{'])
    elif state == "AFTER_OPEN_BRACE":
        valid_ids.append(vocab['"'])
    elif state == "INSIDE_FIRST_KEY":
        valid_ids.append(my_model.encode('"name"').tolist())

while True:
    logits = my_model.get_logits_from_input_ids(mylist)
    # 1. find valid token IDs based on current state
    valid_ids = get_valid_tokens(state, vocab)
    
    # 2. kill invalid tokens
    for i in range(len(logits)):
        if i not in valid_ids:
            logits[i] = float('-inf')
    index_token = int(np.argmax(logits))

    mylist.append(index_token)
    generated_ids.append(index_token)
    if index_token == eos_token_id:
        break

    decoded = my_model.decode([index_token])
    print(f"Generated token: '{decoded}'") 
    if decoded == "{":
        open_brackets += 1
    elif decoded == "}":
        open_brackets -= 1
        if open_brackets == 0:
            break

result = "{" + my_model.decode(generated_ids) + "}"
print(result)


def update_state(state: str, decoded: str) -> str:
    if state == "START" and decoded == "{":
        state = "AFTER_OPEN_BRACE"
    
    elif state == "AFTER_OPEN_BRACE" and decoded == '"':
        state = "INSIDE_FIRST_KEY"
    
    elif state == "INSIDE_FIRST_KEY" and decoded == '"':
        state = "AFTER_KEY"
    
    elif state == "AFTER_KEY" and decoded == ":":
        state = "AFTER_COLON"
    
    elif state == "AFTER_COLON":
        if decoded == '"':
            state == "INSIDE_STRING_VALUE"
        elif decoded.isdigit():
            state == "INSIDE_NUMBER_VALUE"

    elif state == "INSIDE_STRING_VALUE" and decoded == '"':
        state == "AFTER_VALUE"