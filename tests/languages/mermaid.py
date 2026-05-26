import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f8f9fa')

def draw_box(ax, cx, cy, text, color, fontsize=10, width=2.4, height=0.75):
    x, y = cx - width / 2, cy - height / 2
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='#333333',
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4)

def draw_arrow(ax, x0, y0, x1, y1, color, lw=1.8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=25,
                                shrinkA=20, shrinkB=60),
                zorder=2)

nodes = {
    'Spanish':              (8.0, 9.5),
    'Italian':              (8.0, 0.5),
    'Salud Familia':        (5.0, 7.0),
    'Mapfre Salud':         (11.0, 7.0),
    'Wikipedia_IT':         (8.0, 5.0),
    'ISS':                  (5.0, 3.0),
    'Nostro Figlio':        (11.0, 3.0),
    'Contradiction':        (2.5, 5.0),
    'Cultural Discrepancy': (13.5, 5.0),
}

colors = {
    'Spanish':              "#44c02b",
    'Italian':              '#2980b9',
    'Salud Familia':        "#a6b1b1",
    'Mapfre Salud':         '#a6b1b1',
    'Wikipedia_IT':         '#a6b1b1',
    'Nostro Figlio':        '#a6b1b1',
    'ISS':                  '#a6b1b1',
    'Contradiction':        '#8e44ad',
    'Cultural Discrepancy': '#d35400',
}

labels_display = {
    'Wikipedia_IT':  'Wikipedia',
    'Nostro Figlio': 'Nostro Figlio',
    'Salud Familia': 'Salud Familia',
    'Mapfre Salud':  'Mapfre Salud',
    'ISS':           'ISS',
}

for key, (cx, cy) in nodes.items():
    label = labels_display.get(key, key)
    if key in ('Contradiction', 'Cultural Discrepancy'):
        draw_box(ax, cx, cy, label, colors[key], fontsize=22, width=3.8, height=0.85)
    elif key in ('Spanish', 'Italian'):
        draw_box(ax, cx, cy, label, colors[key], fontsize=22, width=2.6, height=0.85)
    else:
        draw_box(ax, cx, cy, label, colors[key], fontsize=22, width=2.8, height=0.75)

sx, sy = nodes['Spanish']
for src in ['Wikipedia_IT', 'Salud Familia', 'Mapfre Salud']:
    draw_arrow(ax, sx, sy, nodes[src][0], nodes[src][1], color='#27ae60', lw=2.0)

ix, iy = nodes['Italian']
for src in ['Wikipedia_IT', 'Nostro Figlio', 'ISS']: 
    draw_arrow(ax, ix, iy, nodes[src][0], nodes[src][1], color='#2980b9', lw=2.0)

contx, conty = nodes['Contradiction']
for src in ['Salud Familia', 'ISS']: #'Wikipedia_IT',
    draw_arrow(ax, nodes[src][0], nodes[src][1], contx, conty, color='#2c3e50', lw=1.8)

dcx, dcy = nodes['Cultural Discrepancy']
for src in ['Nostro Figlio', 'Mapfre Salud']: #'Wikipedia_IT'
    draw_arrow(ax, nodes[src][0], nodes[src][1], dcx, dcy, color='#2c3e50', lw=1.8)

legend_elements = [
    mpatches.Patch(facecolor='#27ae60', label='Spanish sources'),
    mpatches.Patch(facecolor='#2980b9', label='Italian sources'),
    mpatches.Patch(facecolor='#2c3e50', label='→ Label (Contradiction / Cultural Discrepancy)'),
]

ax.annotate('', xy=(contx, conty), xytext=(nodes['Wikipedia_IT'][0], nodes['Wikipedia_IT'][1]),
            arrowprops=dict(arrowstyle='->', color='#2c3e50',
                            lw=1.8, mutation_scale=25,
                            shrinkA=100, shrinkB=140),  # tune shrinkB here
            zorder=5)

ax.annotate('', xy=(dcx, dcy), xytext=(nodes['Wikipedia_IT'][0], nodes['Wikipedia_IT'][1]),
            arrowprops=dict(arrowstyle='->', color='#2c3e50',
                            lw=1.8, mutation_scale=25,
                            shrinkA=100, shrinkB=140),
            zorder=5)
'''
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.9, edgecolor='#cccccc')

ax.set_title('Dataset Sources by Language and Label',
             fontsize=14, fontweight='bold', pad=12, color='#2c3e50')
'''
plt.tight_layout()
plt.savefig('/export/usuarios01/ivgomez/mind/images/mermaid_diagram_final.jpg', dpi=150, bbox_inches='tight')
plt.show()