from transformers import AutoTokenizer, AutoModel
import transformers
import torch

from huggingface_hub import login

login(token="")

model = "meta-llama/Llama-2-7b-hf"
# model = "meta-llama/Llama-2-7b-chat-hf"
# model = "../llama/llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model, low_cpu_mem_usage=True)
pretrained = AutoModel.from_pretrained(model, low_cpu_mem_usage=True)

# tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/hits/basement/nlp/jeonso/.cache/")
# pretrained = AutoModel.from_pretrained(model, cache_dir="/hits/basement/nlp/jeonso/.cache/")

# from transformers import LlamaForCausalLM, LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained("/output/path")
# model = LlamaForCausalLM.from_pretrained("/output/path")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("!!!!")
