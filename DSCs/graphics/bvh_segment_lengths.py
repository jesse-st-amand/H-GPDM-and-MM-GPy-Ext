import numpy as np

class BVHNode:
    def __init__(self, name):
        self.name = name
        self.offsets = np.zeros(3)
        self.children = []

def parse_hierarchy(lines):
    stack = []
    root = None
    current_node = None

    for line in lines:
        if "ROOT" in line or "JOINT" in line:
            parts = line.split()
            name = parts[1]
            new_node = BVHNode(name)
            if not stack:
                root = new_node
            else:
                stack[-1].children.append(new_node)
            stack.append(new_node)
            current_node = new_node
        elif "End Site" in line:
            new_node = BVHNode("End Site")
            stack[-1].children.append(new_node)
            stack.append(new_node)
        elif "{" in line:
            continue
        elif "}" in line:
            stack.pop()
        elif "OFFSET" in line:
            parts = line.split()
            offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            current_node.offsets = offset

    return root

def compute_segment_lengths(node):
    lengths = {}
    for child in node.children:
        length = np.linalg.norm(child.offsets)
        lengths[(node.name, child.name)] = length
        lengths.update(compute_segment_lengths(child))
    return lengths

def read_bvh_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    hierarchy_lines = []
    motion_lines = []
    is_hierarchy = True
    for line in lines:
        if "MOTION" in line:
            is_hierarchy = False
            continue
        if is_hierarchy:
            hierarchy_lines.append(line.strip())
        else:
            motion_lines.append(line.strip())

    root = parse_hierarchy(hierarchy_lines)
    return root

# Example usage:
bvh_file_path = 'D:\\CMU_bvh\\14\\BVH\\elbow_to_knee_01.bvh'
root_node = read_bvh_file(bvh_file_path)
segment_lengths = compute_segment_lengths(root_node)

for segment, length in segment_lengths.items():
    print(f"Length of segment {segment} is {length}")
