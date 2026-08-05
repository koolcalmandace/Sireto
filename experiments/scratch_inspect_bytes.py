with open("src/xgb_matcher/features.py", "rb") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for idx in range(1200, len(lines)):
    print(f"Line {idx+1}: {repr(lines[idx])}")
