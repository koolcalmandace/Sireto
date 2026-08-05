with open("src/xgb_matcher/features.py", "r", encoding="utf-8") as f:
    source = f.read()

parts = source.split('"""')
print(f"Total parts split by triple-quotes: {len(parts)}")
print(f"Total triple-quotes: {len(parts) - 1}")

# Let's see if the count is even. If it is odd, we have an unmatched triple quote!
if (len(parts) - 1) % 2 != 0:
    print("WARNING: Odd number of triple-quotes! There is an unmatched triple quote!")
    # Let's find which part has odd number of quotes or locate it.
    # Let's print the line numbers where each triple quote appears
    lines = source.splitlines()
    quote_lines = []
    for idx, line in enumerate(lines):
        count = line.count('"""')
        for _ in range(count):
            quote_lines.append(idx + 1)
    
    print("Triple-quote line numbers:")
    print(quote_lines)
