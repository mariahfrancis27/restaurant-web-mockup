import anthropic
client = anthropic.Anthropic()
print(client.api_key[:12])  # sanity check it's not truncated/blank
msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say hi"}]
)
print(msg.content)