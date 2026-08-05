import ast

with open("src/xgb_matcher/features.py", "r", encoding="utf-8") as f:
    source = f.read()

try:
    ast.parse(source)
    print("No syntax error found in AST parsing!")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"Line: {e.lineno}")
    print(f"Offset: {e.offset}")
    print(f"Text: {repr(e.text)}")
