import torch
import re
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import numpy as np
import pandas as pd
import random
import validators
import time
import json
import gc
import pynvml
import psutil
import os
import multiprocessing as mp

def clean_up_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def process_prediction_string(s):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    try:
        return s.lower().strip().split()[0].translate(table)
    except IndexError:
       print("bad string:", s)
       return ""

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    ram = psutil.virtual_memory()
    print("RAM usage (%):", ram.percent)
    print("RAM used (GB):", round(ram.used / 1e9, 2))
    print("RAM used by this process (GB):", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

def get_worker(model, input_queue, output_queue, ctx):
    match model:
        case "meta-llama/Llama-3.1-8B-Instruct":
            worker = ctx.Process(
                target=worker_loop_llama31,
                args=(input_queue, output_queue)
            )
        case "meta-llama/Llama-3.2-3B-Instruct":
            worker = ctx.Process(
                target=worker_loop_llama32,
                args=(input_queue, output_queue)
            )
        case "mistralai/Mistral-7B-Instruct-v0.2":
            worker = ctx.Process(
                target=worker_loop_mistral,
                args=(input_queue, output_queue)
            )
        case "Qwen/Qwen3-4B-Instruct-2507":
            worker = ctx.Process(
                target=worker_loop_qwen3,
                args=(input_queue, output_queue)
            )
        case "google/gemma-7b-it":
            worker = ctx.Process(
                target=worker_loop_gemma,
                args=(input_queue, output_queue)
            )
        case "HuggingFaceH4/zephyr-7b-beta":
            worker = ctx.Process(
                target=worker_loop_zephyr,
                args=(input_queue, output_queue)
            )    
        case _:
            raise Exception("model bad")

    worker.start()
    return worker
        

def predict_with_worker(model, urls, batch_size=8):
    ctx = mp.get_context("spawn")  # important for CUDA safety
    input_queue = ctx.Queue(maxsize=4)
    output_queue = ctx.Queue()

    worker = get_worker(model, input_queue, output_queue, ctx)
    
    batch_id = 0
    num_batches = 0

    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        input_queue.put((batch_id, batch))
        batch_id += 1
        num_batches += 1

    results = {}
    for _ in range(num_batches):
        bid, preds = output_queue.get()
        results[bid] = preds

    input_queue.put(None)
    worker.join()

    final_preds = []
    for i in range(num_batches):
        final_preds.extend(results[i])

    return final_preds

def worker_loop_llama31(input_queue, output_queue):
    eightbit_config = BitsAndBytesConfig(load_in_8bit=True)

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    p = re.compile("<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*)<\|eot_id\|>")

    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", quantization_config=eightbit_config)
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        conversations = [[{"role": "user", "content": f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (legitimate/phishing). Do not provide explanations, only the answer."}] for url in urls]
        tokenized = [
            tokenizer.apply_chat_template(
                conv,
                return_tensors="pt",
                add_generation_prompt=True
            )[0]
            for conv in conversations
        ]
        inputs = tokenizer.pad(
            {"input_ids": tokenized},
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        print_gpu_utilization()
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10)
        print_gpu_utilization()
        print(f"batch {batch_id} done")
        preds = []
        for i in range(len(outputs)):
            output = tokenizer.decode(outputs[i], skip_special_tokens=False)
            try:
                m = p.search(output)
                pred = m.group(1)
                pred = re.sub(r"<\|.*?\|>", "", pred)
                preds.append(process_prediction_string(pred))
            except AttributeError:
                preds.append("Error: regex failed")
        del inputs
        del outputs
        gc.collect()
        
        output_queue.put((batch_id, preds))
        
    clean_up_model(model)
    torch.cuda.empty_cache()

def worker_loop_llama32(input_queue, output_queue):
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        prompts = [f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (legitimate/phishing). Do not provide explanations, only the answer." for url in urls]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        print_gpu_utilization()
        generated = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.pad_token_id)
        outputs = [g.cpu() for g in generated]
        del generated
        del inputs
        print_gpu_utilization()
        preds= []
        for i in range(len(outputs)):
            output = tokenizer.decode(outputs[i], skip_special_tokens=True)
            output = output[len(prompts[i]):]
            preds.append(process_prediction_string(output))
        print(f"{batch_id} done")
        
        output_queue.put((batch_id, preds))
        
    clean_up_model(model)
    torch.cuda.empty_cache()

def worker_loop_mistral(input_queue, output_queue):    
    eightbit_config = BitsAndBytesConfig(load_in_8bit=True)
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = MistralTokenizer.v1()
    p = re.compile("\[\/INST\]((\s|.)*)")

    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=eightbit_config)
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        completion_requests = [ChatCompletionRequest(messages=[UserMessage(content=f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (legitimate/phishing). Do not provide explanations, only the answer.")]) for url in urls]
        tokens = [torch.tensor(tokenizer.encode_chat_completion(completion_request).tokens) for completion_request in completion_requests]
        padded = torch.nn.utils.rnn.pad_sequence(
            tokens,
            batch_first=True,
            padding_value=0,
            padding_side="left"
        )
        attention_mask = (padded != 0).long().to("cuda:0")
        padded = padded.to("cuda:0")
        print_gpu_utilization()
        
        preds = []
        generated = model.generate(padded, attention_mask=attention_mask, pad_token_id=0, max_new_tokens=10, do_sample=True)
        for g in generated:
            output = tokenizer.decode(g.tolist())
            try:
                m = p.search(output)
                pred = process_prediction_string(m.group(1))
                preds.append(pred)
            except AttributeError:
                preds.append("Error: regex failed")
        del generated
        del padded
        gc.collect()
        print_gpu_utilization()
        print(f"batch {batch_id} done")
        
        output_queue.put((batch_id, preds))
        
    clean_up_model(model)
    torch.cuda.empty_cache()

def worker_loop_qwen3(input_queue, output_queue):
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda:0"
    )
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        conversations = [[{"role": "user", "content": f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (legitimate/phishing)."}] for url in urls]
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

        print_gpu_utilization()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10
        )
        preds = []
        for g in generated_ids:
            output_ids = g[len(model_inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            print("qwen output:", content)
            preds.append(process_prediction_string(content))
        del generated_ids
        del model_inputs
        print_gpu_utilization()
        print(f"batch {batch_id} done")
        
        output_queue.put((batch_id, preds))
    
    clean_up_model(model)
    torch.cuda.empty_cache()

def worker_loop_gemma(input_queue, output_queue):
    model_name = "google/gemma-7b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        prompts = [f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (phishing or legitimate)." for url in urls]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        print_gpu_utilization()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=20)
        preds = []
        for j in range(len(generated)):
            output = tokenizer.decode(generated[j], skip_special_tokens=True)
            pred = output[len(prompts[j]):]
            preds.append(process_prediction_string(pred))
        del generated
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_utilization()
        print(f"batch {batch_id} done")
        
        output_queue.put((batch_id, preds))
    
    clean_up_model(model)
    torch.cuda.empty_cache()

def worker_loop_zephyr(input_queue, output_queue):
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    p = re.compile("<\|assistant\|>\s*(.*)")
    
    print("model init")
    print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print_gpu_utilization()

    while True:
        item = input_queue.get()

        if item is None:
            break  # shutdown signal

        batch_id, urls = item
        conversations = [[
            {
                "role": "system",
                "content": "You are a LLM assistant tasked with making predictions on whether a website is phishing based only on the URL.",
            },
            {"role": "user", "content": f"Make a prediction whether this is a phishing site's URL: {url}. Answer with 1 word (legitimate or phishing). Do not provide explanations, only the answer."},
        ] for url in urls]
        inputs = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, return_tensors="pt", padding=True).to("cuda")
        print_gpu_utilization()
        with torch.inference_mode():
            generated = model.generate(inputs, max_new_tokens=20)
        preds = []
        for g in generated:
            output = tokenizer.decode(g, skip_special_tokens=False)
            try:
                m = p.search(output)
                pred = m.group(1)
                pred = re.sub(r"<.*?>", "", pred)
                print(pred)
                preds.append(process_prediction_string(pred))
            except AttributeError:
                preds.append("Error: regex failed")
        del generated
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_utilization()
        print(f"batch {batch_id} done")
        
        output_queue.put((batch_id, preds))
    
    clean_up_model(model)
    torch.cuda.empty_cache()

def get_legitimate_urls(filename, sample_size):
    with open(filename, "r") as f:
        text = f.read()
        good_url = [x for x in text.split() if is_valid_url(x)]
        sample = random.sample(good_url, sample_size)
        return sample
    
def get_phishing_urls(filename, sample_size):
  with open(filename, "r") as f:
    text = f.read()
    parsed_json = json.loads(text)
    good_url = [x["url"] for x in parsed_json if is_valid_url(x["url"])]
    sample = random.sample(good_url, sample_size)
    return sample

def is_valid_url(s: str) -> bool:
    if "://" not in s:
        s = "http://" + s
    return validators.url(s)

def evaluate_model(model, dataset):
    labels = [1 if x == "bad" else 0 for x in dataset["Label"].to_list()]
    samples = dataset["URL"].to_list()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    skipped = 0
    preds = predict_with_worker(model, samples, batch_size=64)
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        if pred == "phishing":
            if label == 0:
                    FP += 1
            else:
                    TP += 1
        elif pred == "legitimate":
            if label == 0:
                TN +=1
            else:
                FN += 1
        else:
            print("Error: prediction was", pred)
            skipped += 1
    skipped += len(samples) - len(preds)
    return (TP, FP, FN, TN, skipped)

def test_models(models, dataset_file, results_file):
    dataset = pd.read_csv(dataset_file)
    try:
        results = pd.read_csv(results_file)
    except FileNotFoundError:
        results = pd.DataFrame(
            {
                "model" : [],
                "TP" : [],
                "FP" : [],
                "FN" : [],
                "TN" : [],
                "skipped" : [],
                "accuracy" : [],
                "time" : []
            }
        )
    for model in models:
        try:
            start = time.time()
            tp, fp, fn, tn, skip = evaluate_model(model, dataset)
            end = time.time()
            acc = (tp + tn) / (tp + fp + fn + tn)
            results.loc[len(results)] = {
                "model" : model,
                "TP" : tp,
                "FP" : fp,
                "FN" : fn,
                "TN" : tn,
                "skipped" : skip,
                "accuracy" : acc,
                "time" : end - start
            }
            results.to_csv(results_file, index=False)
        except ZeroDivisionError:
            print("divide by 0")
        except Exception as e:
            print(f"Exception: {e}\nmodel is {model}")

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history()
    models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen3-4B-Instruct-2507", "HuggingFaceH4/zephyr-7b-beta", "google/gemma-7b-it"]
    dataset_file = "test_dataset.csv"
    results_file = "results0.csv"
    start = time.time()
    test_models(models, dataset_file, results_file)
    end = time.time()
    print(f"testing took {end - start} seconds")