import ollama

while True:
    user_input = input("請輸入你的問題，輸入exit離開:")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": user_input}]
    )
    print(response["message"]["content"])
    print("-" * 50)

