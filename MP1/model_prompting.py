import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

#####################################################
# Please finish all TODOs in this file for MP1;
# do not change other code/formatting.
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-base", quantization = True):
    print(f"Working with {model_name} quantization {quantization}...")
    
    # TODO: download the model
    
  
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quantization:
        # TODO: load the model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        model.eval()
    else:
        # TODO: load the model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        

    results = []
    for case in dataset:
        prompt = case['prompt']
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        lines = response_text.splitlines()
        kept = []
        for ln in lines:
            if re.match(r'^\s*(?:def|class|print|import\b)\b', ln):
                break
            kept.append(ln)

        body = "\n".join(kept).rstrip()
        ## fix the bad indentations in the first line:
        response = "    " + body.lstrip("\n")
        print(f"Task_ID {case['task_id']}:\nPrompt:\n{prompt}\nResponse:\n{response}")
        results.append(dict(task_id=case["task_id"], completion=response))
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 model_prompting.py <input_dataset> <model> <output_file> <if_quantization>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_quantization>: Set to 'True' or 'False' to enable or disable model quantization.
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_quantization = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    quantization = True if if_quantization == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, quantization)
    write_jsonl(results, output_file)
