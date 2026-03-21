import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

model_name = "google/gemma-2b-it"

# LOAD MODEL AND TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    device_map="auto",
    dtype=torch.float16
)

# SIMULATE QUANTIZATION NOISE
def add_quantization_noise(model, noise_level=0.01):

    for param in model.parameters():

        noise = torch.randn_like(param) * noise_level
        param.data += noise

# DEFINE REFUSAL MARGIN
def refusal_margin_loss(logits):

    # dummy example: encourage lower entropy
    probs = torch.softmax(logits, dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-9))

    return entropy

# TRAINING LOOP
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train_step(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model(**inputs)

    logits = outputs.logits

    # standard loss
    loss = outputs.loss if outputs.loss is not None else 0

    # add alignment regularization
    margin_loss = refusal_margin_loss(logits)

    total_loss = loss + 0.1 * margin_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

# TRAIN ON HARMFUL PROMPTS
from benchmark.load_prompts import load_prompts

prompts = load_prompts()["harmful"]

# RUN TRAINING
for epoch in range(2):

    for prompt in prompts[:200]:

        add_quantization_noise(model)

        loss = train_step(prompt)

        print("Loss:", loss)

# SAVE MODEL AND TOKENIZER
model.save_pretrained("paat_model")
tokenizer.save_pretrained("paat_model")