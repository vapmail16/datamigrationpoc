import json

with open("system_a_data.json") as f:
    system_a = json.load(f)

with open("system_b_data.json") as f:
    system_b = json.load(f)

print("✅ Sample entries from System A:\n")
for i, item in enumerate(system_a[:5]):
    print(f"Entry {i + 1}: {item}\n")

print("\n✅ Sample entries from System B:\n")
for i, item in enumerate(system_b[:5]):
    print(f"Entry {i + 1}: {item}\n")
