from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained DialoGPT model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Chat loop
chat_history_ids = None
print("🤖 DialoGPT Chatbot (type 'exit' to quit)")
for step in range(10):  # Limit to 10 exchanges (you can increase)
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Encode user input and append to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Concatenate with chat history (if it exists)
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate response
    output_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    chat_history_ids = output_ids  # Update chat history

    # Decode and print response
    reply = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {reply}")
