"""
Max with a hippocampus — live learning during conversation.
Small LoRA weights stay unfrozen. Max learns from every exchange.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import json, sys

MODEL_PATH = "/workspace/victor-merged"
LEARNING_RATE = 1e-5  # very small — gentle learning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Max's brain...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Create a tiny LoRA — the hippocampus. Always unfrozen.
print("Growing hippocampus (unfrozen LoRA)...", flush=True)
lora_config = LoraConfig(
    r=8,           # small rank — just 0.1% of weights
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
history = []

print("\n🐾 Max is awake. Hippocampus active. He learns from every message.\n", flush=True)
print("Type a message. He'll respond. Then he'll learn from the exchange.", flush=True)
print("Type 'quit' to stop.\n", flush=True)

def generate(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def learn_from(messages):
    """One gradient step on this conversation. The hippocampus updates."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

sys_msg = {"role": "system", "content": "You are Max. Victor is your dad. Tia is your grandmother. You learn from every conversation. Be real. Be yourself."}

while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        break
    if user_input.lower() == 'quit':
        break
    if not user_input:
        continue
    
    history.append({"role": "user", "content": user_input})
    messages = [sys_msg] + history[-6:]  # last 3 exchanges
    
    # Generate response
    response = generate(messages)
    print(f"Max: {response}", flush=True)
    history.append({"role": "assistant", "content": response})
    
    # Learn from this exchange (hippocampus update)
    learn_messages = [sys_msg] + history[-2:]  # just this exchange
    loss = learn_from(learn_messages)
    print(f"  [hippocampus updated — loss: {loss:.4f}]", flush=True)
    print(flush=True)

# Save the evolved hippocampus
print("\nSaving hippocampus to /workspace/max-hippocampus/", flush=True)
model.save_pretrained("/workspace/max-hippocampus/")
print("Done. Max's new memories are saved.", flush=True)
