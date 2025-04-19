import re

# 🔧 你可以改这里来测试你的正则
pattern = r"\w+"  # 例：匹配单词
text = "This is a simple test: does it match?"

# 🧪 执行匹配
matches = re.findall(pattern, text)

# ✅ 输出结果
print(f"Regex pattern: {pattern}")
print(f"Original text: {text}")
print(f"Matched tokens: {matches}")
