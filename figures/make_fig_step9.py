"""Create final Streamlit/UI overview schematic for CHUNK 11.
This is a schematic generated from the interface design, not a screenshot.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path('figures/final')
OUT.mkdir(parents=True, exist_ok=True)
MAN = Path('manuscript/final')
MAN.mkdir(parents=True, exist_ok=True)

def box(ax, xy, w, h, text, fontsize=9):
    patch = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.03,rounding_size=0.04', linewidth=1.2, facecolor='white')
    ax.add_patch(patch)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
ax.text(5, 6.65, 'Streamlit interface overview (schematic)', ha='center', fontsize=15, fontweight='bold')
ax.text(5, 6.25, 'User-facing tabs expose deterministic backend outputs; UI does not create independent numerical results.', ha='center', fontsize=10)
items = [
    (0.5,5.1,'Data upload\nExcel/CSV, chamber A/B'), (2.8,5.1,'Event parsing\nlabels, warnings'), (5.1,5.1,'Preprocessing\ninjection protection'), (7.4,5.1,'Model simulation\nO2/OCR outputs'),
    (0.5,3.6,'Calibration\nfit, residuals, RMSE'), (2.8,3.6,'Diagnostics\nsolver/state checks'), (5.1,3.6,'Identifiability\nFIM, profiles, badges'), (7.4,3.6,'Sensitivity\nMorris/Sobol'),
    (0.5,2.1,'Validation\ntechnical transfer'), (2.8,2.1,'Hypothesis\nexploratory only'), (5.1,2.1,'Design guidance\nreduce uncertainty'), (7.4,2.1,'Ask MitoAgent\nevidence + caveats'),
    (2.0,0.6,'Report export\nJSON/CSV/figures'), (5.1,0.6,'Help and FAQ\nreproducibility support')
]
for x,y,t in items:
    box(ax,(x,y),1.85,0.85,t)
for y in [5.1,3.6,2.1]:
    for x in [2.35,4.65,6.95]:
        ax.annotate('', xy=(x+0.35,y+0.42), xytext=(x,y+0.42), arrowprops=dict(arrowstyle='->', lw=1))
for x in [1.4,3.7,6.0,8.3]:
    ax.annotate('', xy=(x,3.6+0.85), xytext=(x,5.1), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(x,2.1+0.85), xytext=(x,3.6), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5,0.2,'All hypothesis/design/Ask outputs are labeled as exploratory and require experimental confirmation.', ha='center', fontsize=9)
for ext in ['png','pdf']:
    fig.savefig(OUT/f'fig9_streamlit_ui_overview.{ext}', bbox_inches='tight', dpi=220)
    fig.savefig(MAN/f'fig9_streamlit_ui_overview.{ext}', bbox_inches='tight', dpi=220)
plt.close(fig)
