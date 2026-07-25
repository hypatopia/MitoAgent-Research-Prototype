"""Create final Ask MitoAgent / optional LLM role schematic for CHUNK 11."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path('figures/final'); OUT.mkdir(parents=True, exist_ok=True)
MAN = Path('manuscript/final'); MAN.mkdir(parents=True, exist_ok=True)

def box(ax, xy, w, h, text, fontsize=9):
    patch = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.03,rounding_size=0.04', linewidth=1.2, facecolor='white')
    ax.add_patch(patch)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis('off')
ax.text(5, 6.15, 'Ask MitoAgent interpretation pathway', ha='center', fontsize=15, fontweight='bold')
ax.text(5, 5.78, 'Natural-language answers are generated only from structured deterministic backend outputs.', ha='center', fontsize=10)
box(ax,(0.5,4.6),2.0,0.9,'User question\n“Can this trace suggest\nComplex IV dysfunction?”')
box(ax,(3.0,4.6),2.0,0.9,'Question router\nclassifies intent and\nrequired evidence')
box(ax,(5.5,4.6),2.0,0.9,'Safety rules\nunsupported disease /\nmechanism claims blocked')
box(ax,(8.0,4.6),1.5,0.9,'Answer mode\ndeterministic or\nLLM-assisted')
for x in [2.55,5.05,7.55]:
    ax.annotate('', xy=(x+0.35,5.05), xytext=(x,5.05), arrowprops=dict(arrowstyle='->', lw=1))

backend = ['Parsed events','Phase OCR summary','Calibration','Numerical diagnostics','Identifiability','Sensitivity','Validation','Warnings']
for i,t in enumerate(backend):
    x = 0.7 + (i%4)*2.25; y = 3.05 - (i//4)*0.85
    box(ax,(x,y),1.75,0.55,t,fontsize=8)
ax.text(5,3.85,'Backend evidence used', ha='center', fontsize=11, fontweight='bold')
ax.annotate('', xy=(5,4.55), xytext=(5,3.65), arrowprops=dict(arrowstyle='<->', lw=1))

box(ax,(0.7,0.8),2.4,1.0,'Caveat labels\nExploratory only\nOCR-only limitation applies')
box(ax,(3.8,0.8),2.4,1.0,'Candidate hypotheses\nrequiring experimental\nconfirmation')
box(ax,(6.9,0.8),2.4,1.0,'Recommended next action\nrun missing diagnostics or\nadd follow-up measurement')
for x in [2.3,5.0,7.5]:
    ax.annotate('', xy=(x,1.85), xytext=(x,2.45), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5,0.25,'The optional LLM does not estimate parameters, generate figures, override diagnostics, or produce scientific results.', ha='center', fontsize=9)
for ext in ['png','pdf']:
    fig.savefig(OUT/f'fig10_ask_mitoagent.{ext}', bbox_inches='tight', dpi=220)
    fig.savefig(MAN/f'fig10_ask_mitoagent.{ext}', bbox_inches='tight', dpi=220)
plt.close(fig)
