# 因为V-ARC数据不同所以把它转换成和别的相同的

def parse_harc_grid(grid_str: str):
    """
    Example:
    '|233|471|137|462|' -> [[2,3,3],[4,7,1],[1,3,7],[4,6,2]]
    """
    rows = [row for row in grid_str.strip().split("|") if row]
    grid = [[int(ch) for ch in row] for row in rows]
    return grid


example = "|233|471|137|462|"
parsed = parse_harc_grid(example)

print("Original string:")
print(example)
print()

print("Parsed grid:")
print(parsed)