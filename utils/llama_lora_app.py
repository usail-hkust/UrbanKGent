from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
import uvicorn, json, datetime
import torch
import transformers
from peft import PeftModel


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI()
B_INST, E_INST = "[INST]", "[/INST]"

model = None
tokenizer = None
last_model_name = None
init = False
gpu_count = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words: list, tokenizer):
        self.keywords = [torch.LongTensor(tokenizer.encode(w)[-5:]).to(device) for w in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if len(input_ids[0]) >= len(k) and torch.equal(input_ids[0][-len(k):], k):
                return True
        return False


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer, last_model_name, gpu_count, init
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    add_lora = json_post_list.get('add_lora')
    stop = json_post_list.get('stop')
    temperature = json_post_list.get('temperature')

    # init model
    if not init:
        model_path = "../data/llm_models/llama-2-7b-chat-hf"
        lora_path = "../data/llm_models/llama-2-7b-finetune/"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = {}

        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        tokenizer.add_special_tokens(special_tokens_dict)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            # load_in_8bit=True,
            device_map="auto",
        )

        if add_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float16,
            )
        print("====================================")
        print(f"{model_path}\t{lora_path} Load success!")
        init = True

    encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, tokenizer)]) if stop else None
    generated_ids = model.generate(
        input_ids=encoded_prompt["input_ids"],
        max_new_tokens=2048,
        do_sample=True,
        early_stopping=False,
        num_return_sequences=1,
        temperature=temperature if temperature else 0.1,
        top_p=1.0,
        top_k=50,
        stopping_criteria=stop_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    ).to(device)
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if stop:
        for ending in stop:
            if decoded_output[0].endswith(ending):
                decoded_output[0] = decoded_output[0][:-len(ending)]
                break

    response = decoded_output[0][len(prompt):]

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer
