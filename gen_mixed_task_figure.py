"""Generate the Mixed-Task Inventory Preflight figure as SVG and PNG."""
import math

# === Configuration ===
W, H = 1600, 900
PANEL_W = 620
LEFT_X = 80
RIGHT_X = W - PANEL_W - 80
CENTER_X = W // 2

# Colors
BG = "#FAFAFA"
PANEL_BG = "#FFFFFF"
PANEL_BORDER = "#E0E0E0"

# Task colors (muted)
TASK_COLORS = {
    "QNLI": "#5B8DBE",  # muted blue
    "RTE":  "#6BAF7A",   # muted green
    "MRPC": "#C27C5E",   # muted terracotta
    "SST-2":"#9B7DBF",   # muted purple
}
TASK_COLORS_LIGHT = {
    "QNLI": "#D6E4F0",
    "RTE":  "#D4EDDA",
    "MRPC": "#F0DDD2",
    "SST-2":"#E3D8F0",
}

GRAY_LINE = "#C8C8C8"
GRAY_LINE_DARK = "#AAAAAA"
CAUTION_COLOR = "#E8A838"
CAUTION_BG = "#FFF5E0"
SAFE_ACCENT = "#4A90B8"
SAFE_BG = "#EBF3FA"
RETAINED_COLOR = "#2B7A3E"
RETAINED_STROKE = "#1D6B2F"
TEXT_DARK = "#2C2C2C"
TEXT_MED = "#5A5A5A"
TEXT_LIGHT = "#888888"

# Adapters: (name, task, left_pos, right_pos)
# Left panel: arranged in a 2x4 grid-ish layout within the panel
# Right panel: same positions but grouped into zones

tasks = ["QNLI", "RTE", "MRPC", "SST-2"]
adapters = []
for i, task in enumerate(tasks):
    adapters.append({"name": f"{task}-A", "task": task, "idx": i * 2})
    adapters.append({"name": f"{task}-B", "task": task, "idx": i * 2 + 1})

# Positions for left panel (ring layout)
def ring_positions(cx, cy, r, n):
    """Return n positions arranged in a ring."""
    positions = []
    for i in range(n):
        angle = -math.pi / 2 + (2 * math.pi * i / n)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        positions.append((x, y))
    return positions

# Left panel center
lcx = LEFT_X + PANEL_W // 2
lcy = H // 2 + 20
left_positions = ring_positions(lcx, lcy, 170, 8)

# Right panel: group by task in 2x2 grid of clusters
rcx = RIGHT_X + PANEL_W // 2
rcy = H // 2 + 20

# Cluster centers (2x2 grid)
cluster_centers = [
    (rcx - 140, rcy - 110),  # QNLI top-left
    (rcx + 140, rcy - 110),  # RTE top-right
    (rcx - 140, rcy + 110),  # MRPC bottom-left
    (rcx + 140, rcy + 110),  # SST-2 bottom-right
]

# Within each cluster, two adapters side by side
right_positions = []
for ci, (cx, cy) in enumerate(cluster_centers):
    right_positions.append((cx - 40, cy))
    right_positions.append((cx + 40, cy))

# All pairs
all_pairs = []
for i in range(8):
    for j in range(i + 1, 8):
        all_pairs.append((i, j))

# Same-task pairs (retained candidates)
same_task_pairs = [(i, i + 1) for i in range(0, 8, 2)]  # 4 pairs

# Cross-task pairs
cross_task_pairs = [p for p in all_pairs if p not in same_task_pairs]

# Build SVG
svg_parts = []

def add(s):
    svg_parts.append(s)

add(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
  <filter id="shadow" x="-2%" y="-2%" width="104%" height="104%">
    <feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="#00000018"/>
  </filter>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="{TEXT_MED}" />
  </marker>
</defs>
<style>
  text {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
</style>
''')

# Background
add(f'<rect width="{W}" height="{H}" fill="{BG}" rx="0"/>')

# === FIGURE TITLE ===
add(f'<text x="{W//2}" y="42" text-anchor="middle" font-size="22" font-weight="600" fill="{TEXT_DARK}">Mixed-Task Inventory Preflight</text>')

# === LEFT PANEL ===
panel_y = 62
panel_h = H - 130

add(f'<rect x="{LEFT_X}" y="{panel_y}" width="{PANEL_W}" height="{panel_h}" rx="12" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" stroke-width="1" filter="url(#shadow)"/>')

# Left panel header
add(f'<text x="{lcx}" y="{panel_y + 35}" text-anchor="middle" font-size="17" font-weight="600" fill="{TEXT_DARK}">Before preflight</text>')
add(f'<text x="{lcx}" y="{panel_y + 55}" text-anchor="middle" font-size="12" fill="{TEXT_LIGHT}">Flat mixed-task candidate space</text>')

# Draw all pair connections (left panel) — gray, varying weight for density feel
for i, j in all_pairs:
    x1, y1 = left_positions[i]
    x2, y2 = left_positions[j]
    # Vary width slightly for visual texture
    w = 1.2 if (i + j) % 3 == 0 else 0.8
    op = 0.55 if (i + j) % 3 == 0 else 0.35
    add(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{GRAY_LINE_DARK}" stroke-width="{w}" opacity="{op}"/>')

# Draw adapter nodes (left panel)
NODE_R = 28
for idx, adapter in enumerate(adapters):
    x, y = left_positions[idx]
    color = TASK_COLORS[adapter["task"]]
    light = TASK_COLORS_LIGHT[adapter["task"]]
    add(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{NODE_R}" fill="{light}" stroke="{color}" stroke-width="2"/>')
    # Short label
    label = adapter["name"]
    add(f'<text x="{x:.0f}" y="{y + 1:.0f}" text-anchor="middle" dominant-baseline="middle" font-size="10" font-weight="600" fill="{color}">{label}</text>')

# Left panel annotations
add(f'<text x="{LEFT_X + 20}" y="{panel_y + panel_h - 20}" font-size="12" fill="{TEXT_LIGHT}">8 adapters · 28 possible pairs</text>')

# === CENTER TRANSITION ===
arrow_y = H // 2 + 10
arrow_x1 = LEFT_X + PANEL_W + 18
arrow_x2 = RIGHT_X - 18
arrow_mid = (arrow_x1 + arrow_x2) / 2

add(f'<line x1="{arrow_x1}" y1="{arrow_y}" x2="{arrow_x2 - 8}" y2="{arrow_y}" stroke="{TEXT_MED}" stroke-width="2" marker-end="url(#arrowhead)"/>')
add(f'<text x="{arrow_mid}" y="{arrow_y - 20}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_DARK}">Gradience</text>')
add(f'<text x="{arrow_mid}" y="{arrow_y - 6}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_DARK}">preflight</text>')
add(f'<text x="{arrow_mid}" y="{arrow_y + 28}" text-anchor="middle" font-size="9" fill="{TEXT_LIGHT}">QA + pair reports +</text>')
add(f'<text x="{arrow_mid}" y="{arrow_y + 40}" text-anchor="middle" font-size="9" fill="{TEXT_LIGHT}">task-boundary advisory</text>')

# === RIGHT PANEL ===
add(f'<rect x="{RIGHT_X}" y="{panel_y}" width="{PANEL_W}" height="{panel_h}" rx="12" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" stroke-width="1" filter="url(#shadow)"/>')

# Right panel header
add(f'<text x="{rcx}" y="{panel_y + 35}" text-anchor="middle" font-size="17" font-weight="600" fill="{TEXT_DARK}">After preflight</text>')
add(f'<text x="{rcx}" y="{panel_y + 55}" text-anchor="middle" font-size="12" fill="{TEXT_LIGHT}">Smaller, more defensible evaluation subset</text>')

# Draw same-task safe zone backgrounds
ZONE_W = 130
ZONE_H = 80
for ci, (cx, cy) in enumerate(cluster_centers):
    add(f'<rect x="{cx - ZONE_W//2}" y="{cy - ZONE_H//2}" width="{ZONE_W}" height="{ZONE_H}" rx="10" fill="{SAFE_BG}" stroke="{SAFE_ACCENT}" stroke-width="1.5" opacity="0.7"/>')
    # Task label above cluster
    task = tasks[ci]
    add(f'<text x="{cx}" y="{cy - ZONE_H//2 - 10}" text-anchor="middle" font-size="11" fill="{TASK_COLORS[task]}" font-weight="600">{task}</text>')

# Draw cross-task caution connections (faded, dashed) — show ~8 representative lines
shown = 0
for i, j in cross_task_pairs:
    x1, y1 = right_positions[i]
    x2, y2 = right_positions[j]
    if (i + j) % 3 == 0 or (i * j) % 5 == 1:
        if shown < 8:
            add(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{CAUTION_COLOR}" stroke-width="1.2" stroke-dasharray="5,4" opacity="0.4"/>')
            shown += 1

# Draw retained same-task connections (bold, green)
for i, j in same_task_pairs:
    x1, y1 = right_positions[i]
    x2, y2 = right_positions[j]
    add(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{RETAINED_COLOR}" stroke-width="3.5" opacity="0.9" stroke-linecap="round"/>')

# Draw adapter nodes (right panel)
for idx, adapter in enumerate(adapters):
    x, y = right_positions[idx]
    color = TASK_COLORS[adapter["task"]]
    light = TASK_COLORS_LIGHT[adapter["task"]]
    add(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{NODE_R}" fill="{light}" stroke="{color}" stroke-width="2"/>')
    label = adapter["name"]
    add(f'<text x="{x:.0f}" y="{y + 1:.0f}" text-anchor="middle" dominant-baseline="middle" font-size="10" font-weight="600" fill="{color}">{label}</text>')

# Right panel legend
legend_x = RIGHT_X + 18
legend_y = panel_y + panel_h - 60

add(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{RETAINED_COLOR}" stroke-width="3"/>')
add(f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="11" fill="{TEXT_MED}">retained candidate (same-task safe zone)</text>')

add(f'<line x1="{legend_x}" y1="{legend_y + 20}" x2="{legend_x + 24}" y2="{legend_y + 20}" stroke="{CAUTION_COLOR}" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>')
add(f'<text x="{legend_x + 32}" y="{legend_y + 24}" font-size="11" fill="{TEXT_MED}">cross-task caution (deprioritized)</text>')

# Right panel annotation
add(f'<text x="{RIGHT_X + 20}" y="{panel_y + panel_h - 20}" font-size="12" font-weight="500" fill="{RETAINED_COLOR}">4 retained candidates</text>')

# === FOOTER CAPTION ===
add(f'<text x="{W//2}" y="{H - 22}" text-anchor="middle" font-size="13" fill="{TEXT_MED}">Example outcome: 28 possible pairs → 4 retained candidates</text>')

add('</svg>')

svg_content = "\n".join(svg_parts)

# Write SVG
svg_path = "/sessions/brave-intelligent-galileo/mnt/gradience/mixed_task_inventory_preflight.svg"
with open(svg_path, "w") as f:
    f.write(svg_content)
print(f"SVG written to {svg_path}")

# Export PNG
import cairosvg
png_path = svg_path.replace(".svg", ".png")
cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=1600, output_height=900)
print(f"PNG written to {png_path}")
