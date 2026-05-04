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
# open_brackets = 0
eos_token_id = my_model.encode("<|endoftext|>").tolist()[0]

generated_ids = []

vocab_path = my_model.get_path_to_vocab_file()

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# invert it to get id → token
id_to_token = {v: k for k, v in vocab.items()}
state = "START"
function_name_ids = []
selected_function = None
current_key_ids = []
current_key = None
expected_type = None
after_name_value = False

def get_valid_tokens(state: str, vocab: dict, expected_type) -> list:
    valid_ids = []
    if state == "START":
        valid_ids.append(vocab['{'])
    elif state == "AFTER_OPEN_BRACE":
        valid_ids.append(vocab['"'])
    elif state == "INSIDE_FIRST_KEY":
        valid_ids.extend(my_model.encode('"name"')[0].tolist())
    elif state == "AFTER_KEY":
        valid_ids.append(vocab[':'])


    elif state == "AFTER_COLON":
        if expected_type == "string":
            valid_ids.append(vocab['"'])
        elif expected_type == "number":
            for token in vocab.keys():
                if token.isdigit():
                    valid_ids.append(vocab[token])
        elif expected_type == "boolean":
            valid_ids.append(vocab['f'])
            valid_ids.append(vocab['t'])
        elif expected_type == "object":
            valid_ids.append(vocab['{'])

    elif state == "INSIDE_STRING_VALUE":
        for token in vocab.keys():
            if not any(c in token for c in ['\n', '\r', '}', '{', ':']):
                valid_ids.append(vocab[token])
    elif state == "INSIDE_NUMBER_VALUE":
            for token in vocab.keys():
                if all(c.isdigit() or c == '.' for c in token):
                    valid_ids.append(vocab[token])
            valid_ids.append(vocab[','])
            valid_ids.append(vocab['}'])
    elif state == "AFTER_VALUE":
        valid_ids.append(vocab[','])
        valid_ids.append(vocab['}'])

    elif state == "INSIDE_SECOND_KEY":
        valid_ids.extend(my_model.encode('"parameters"')[0].tolist())

    elif state == "INSIDE_KEY":
        for token in vocab.keys():
            if all(c.isalnum() or c == '_' for c in token):
                valid_ids.append(vocab[token])
        valid_ids.append(vocab['"'])

    return valid_ids

def update_state(state: str, decoded: str, after_name_value: bool) -> tuple:
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
            state = "INSIDE_STRING_VALUE"
        elif decoded.isdigit():
            state = "INSIDE_NUMBER_VALUE"
        elif decoded == '{':
            state = "AFTER_OPEN_BRACE"

    elif state == "INSIDE_STRING_VALUE" and decoded == '"':
        state = "AFTER_VALUE"

    elif state == "AFTER_VALUE":
        if decoded == ",":
            if not after_name_value:
                state = "INSIDE_SECOND_KEY"
                after_name_value = True
            else:
                state = "INSIDE_KEY"
        elif decoded == '}':
            state = "DONE"

    elif state == "INSIDE_SECOND_KEY" and decoded == '"':
        state = "AFTER_KEY"

    elif state == "INSIDE_NUMBER_VALUE":
        if decoded == ',':
            state = "INSIDE_KEY"
        elif decoded == '}':
            state = "DONE"

    elif state == "INSIDE_KEY" and decoded == '"':
        state = "AFTER_KEY"
    return state, after_name_value

while True:
    logits = my_model.get_logits_from_input_ids(mylist)
    # 1. find valid token IDs based on current state
    valid_ids = get_valid_tokens(state, vocab, expected_type)
    
    # 2. kill invalid tokens
    for i in range(len(logits)):
        if i not in valid_ids:
            logits[i] = float('-inf')
    index_token = int(np.argmax(logits))
    decoded = my_model.decode([index_token])

    mylist.append(index_token)
    generated_ids.append(index_token)

    if state == "INSIDE_FIRST_KEY":
        current_key = "name"

    if state == "INSIDE_SECOND_KEY":
        current_key = "parameters"

    if state == "AFTER_VALUE" or state == "AFTER_OPEN_BRACE" or (state == "INSIDE_SECOND_KEY" and decoded == '"'):
        current_key_ids = []

    if state == "INSIDE_KEY":
        current_key_ids.append(index_token)

    if state == "AFTER_KEY":
        current_key = my_model.decode(current_key_ids)

    if state == "INSIDE_STRING_VALUE" and selected_function is None:
        function_name_ids.append(index_token)
    
    if state == "AFTER_VALUE" and selected_function is None:
        function_name = my_model.decode(function_name_ids)
        selected_function = next(f for f in functions_definition if f['name'] == function_name)
        param_names = list(selected_function["parameters"].keys())
        param_types = {k: v["type"] for k, v in selected_function["parameters"].items()}

    if state == "AFTER_COLON":
        if current_key == "name":
            expected_type = "string"
        elif current_key == "parameters":
            expected_type = "object"
        elif selected_function and current_key in param_types:
            expected_type = param_types[current_key]

    if index_token == eos_token_id:
        break


    print(f"Generated token: '{decoded}'") 
    # if decoded == "{":
    #     open_brackets += 1
    # elif decoded == "}":
    #     open_brackets -= 1
    #     if open_brackets == 0:
    #         break
    state, after_name_value = update_state(state, decoded, after_name_value)
    if state == "DONE":
        break
result = "{" + my_model.decode(generated_ids) + "}"
print(result)

