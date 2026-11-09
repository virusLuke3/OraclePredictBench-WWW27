"""Number-free schematic of executed oracle evaluation procedures.

Restore the user's chosen two-row layout, green group containers, blue check
containers, black line icons, and sparse dashed links. Only claims are corrected.
All icons are original vector line drawings; no screenshot or raster asset is used.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc, Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle,
)
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper/figures'


def draw():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    ax.set(xlim=(0, 133), ylim=(0, 38))
    ax.axis('off')
    black = '#131a20'
    green = '#dff1df'
    blue = '#d8e7fb'
    labels = []

    def text(x, y, value, size=8.0, ha='center', italic=False, bold=False, rotation=0):
        labels.append(value)
        ax.text(x, y, value, ha=ha, va='center', fontsize=size, color=black,
                fontstyle='italic' if italic else 'normal',
                fontweight='bold' if bold else 'normal',
                rotation=rotation, linespacing=1.2, zorder=5)

    def container(x, y, w, h, color, rounding=2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rounding}',
            facecolor=color, edgecolor='none', zorder=0))

    def arrow(x1, y1, x2, y2, both=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='<|-|>' if both else '-|>',
            mutation_scale=10, linewidth=1.3, color=black,
            shrinkA=0, shrinkB=0, zorder=3))

    def icon(kind, x, y, size=4.4):
        """Draw consistently weighted, monochrome icons in a local unit square."""
        transform = Affine2D().scale(size).translate(x-size/2, y-size/2) + ax.transData

        def line(points, lw=1.15):
            path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
            ax.add_patch(PathPatch(path, transform=transform, fill=False,
                edgecolor=black, lw=lw, capstyle='round', joinstyle='round', zorder=4))

        def rect(x, y, w, h, rounded=False):
            patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.06') \
                if rounded else Rectangle((x, y), w, h)
            patch.set(transform=transform, facecolor='none', edgecolor=black, lw=1.15, zorder=4)
            ax.add_patch(patch)

        def circle(x, y, r, filled=False):
            ax.add_patch(Circle((x, y), r, transform=transform,
                facecolor=black if filled else 'none', edgecolor=black, lw=1.15, zorder=4))

        def arc(x, y, w, h, a, b):
            ax.add_patch(Arc((x, y), w, h, theta1=a, theta2=b,
                            transform=transform, color=black, lw=1.15, zorder=4))

        if kind == 'records':
            rect(.20, .08, .60, .75, True)
            rect(.35, .76, .30, .16, True)
            for y in [.61, .43, .25]:
                line([(.32, y), (.68, y)])
        elif kind == 'model':
            rect(.40, .70, .22, .22)
            line([(.51, .70), (.51, .48), (.22, .48), (.22, .29)])
            line([(.51, .48), (.80, .48), (.80, .29)])
            rect(.11, .07, .22, .22)
            rect(.69, .07, .22, .22)
        elif kind == 'llm':
            rect(.15, .25, .70, .52, True)
            circle(.36, .55, .055)
            circle(.64, .55, .055)
            line([(.35, .37), (.65, .37)])
            line([(.5, .77), (.5, .88)])
            circle(.5, .94, .045)
            line([(.22, .25), (.16, .10), (.84, .10), (.78, .25)])
        elif kind == 'tools':
            # A wrench and a short screwdriver, with rounded line ends.
            line([(.12, .86), (.14, .66), (.35, .56), (.74, .14)])
            line([(.12, .86), (.29, .73), (.42, .85), (.40, .97)])
            line([(.40, .97), (.56, .77), (.47, .57)])
            line([(.20, .13), (.68, .66)])
            line([(.63, .72), (.79, .90), (.91, .79), (.74, .61), (.63, .72)])
            circle(.77, .12, .065)
        elif kind == 'forecast':
            # Probability/forecast icon: a dial with a needle, without numerals.
            arc(.5, .43, .78, .78, 0, 180)
            line([(.11, .43), (.11, .20), (.89, .20), (.89, .43)])
            line([(.5, .43), (.72, .68)])
            circle(.5, .43, .055, True)
            for points in [[(.21, .62), (.26, .58)], [(.5, .82), (.5, .75)],
                           [(.79, .62), (.74, .58)]]:
                line(points)
        elif kind == 'calendar':
            rect(.12, .12, .76, .69, True)
            line([(.12, .61), (.88, .61)])
            line([(.31, .73), (.31, .94)])
            line([(.69, .73), (.69, .94)])
            line([(.37, .18), (.37, .54)])
            line([(.63, .18), (.63, .54)])
        elif kind == 'outcome':
            line([(.50, .13), (.50, .85)])
            line([(.17, .72), (.83, .72)])
            line([(.20, .72), (.06, .40), (.34, .40), (.20, .72)])
            line([(.80, .72), (.66, .40), (.94, .40), (.80, .72)])
            arc(.2, .40, .28, .23, 180, 360)
            arc(.8, .40, .28, .23, 180, 360)
            line([(.28, .12), (.72, .12)])
        elif kind == 'interval':
            line([(.12, .54), (.88, .54)])
            line([(.12, .34), (.12, .74)])
            line([(.88, .34), (.88, .74)])
            circle(.52, .54, .075, True)
        elif kind == 'clock':
            circle(.5, .46, .36)
            line([(.5, .46), (.5, .67)])
            line([(.5, .46), (.70, .37)])
            line([(.5, .82), (.5, .94)])
            line([(.38, .94), (.62, .94)])
        elif kind == 'results':
            rect(.18, .08, .64, .77, True)
            rect(.35, .78, .30, .14, True)
            line([(.30, .57), (.42, .47), (.56, .62), (.70, .54)])
            line([(.31, .33), (.68, .33)])
            line([(.31, .20), (.57, .20)])
        elif kind == 'audit':
            rect(.14, .12, .56, .77, True)
            line([(.27, .71), (.55, .71)])
            line([(.27, .55), (.44, .55)])
            circle(.66, .32, .20)
            line([(.81, .17), (.96, .02)])
        elif kind == 'score':
            # Neutral loss/interval icon; no upward trend or success marker.
            line([(.12, .18), (.12, .86)])
            for y, left, right, point in [(.71,.29,.87,.57), (.39,.35,.80,.49)]:
                line([(left,y),(right,y)])
                line([(left,y-.07),(left,y+.07)])
                line([(right,y-.07),(right,y+.07)])
                circle(point,y,.035,True)
        elif kind == 'comparison':
            circle(.43, .60, .30)
            line([(.64, .38), (.92, .09)])
            line([(.29, .68), (.57, .68)])
            line([(.29, .52), (.57, .52)])
        else:
            raise ValueError(kind)

    # The dividing rules and row labels carry the hierarchy without headings.
    ax.plot([4.4, 4.4], [.6, 37.0], color=black, lw=.65)
    ax.plot([.8, 132], [13, 13], color=black, lw=.65)
    text(1.4, 26, 'Workflow', size=8.0, bold=True, rotation=90)
    text(1.4, 6.4, 'Checks', size=8.0, bold=True, rotation=90)

    # Sparse, left-to-right operational flow.
    icon('records', 16.5, 30.3, 5.7)
    text(16.5, 23.7, 'Proposal-time\nfeatures', italic=True, size=8.1)

    container(27, 16.6, 28.5, 20, green, rounding=3.2)
    text(41.25, 33.7, 'Prediction methods', italic=True, size=8.3)
    for y, kind, label in [
        (29.0, 'model', 'Conventional models'),
        (24.5, 'llm', 'Direct LLM'),
        (20.0, 'tools', 'Tool-using LLM'),
    ]:
        icon(kind, 30.8, y, 3.5)
        text(34.1, y, label, ha='left', size=7.25)

    icon('forecast', 67.5, 30.5, 5.9)
    text(67.5, 23.7, 'Disagreement\nforecast', italic=True, size=8.1)

    container(78.0, 17.1, 35.2, 19, green, rounding=3.0)
    text(95.25, 33.5, 'Retrospective evaluation', italic=True, size=8.0)
    for x, y, kind, label in [
        (81.7, 28.3, 'calendar', 'Temporal\nsplit'),
        (97.9, 28.3, 'outcome', 'Recorded\noutcomes'),
        (81.7, 21.9, 'interval', 'Paired\nintervals'),
        (97.9, 21.9, 'clock', 'Warm-run\nlatency'),
    ]:
        icon(kind, x, y, 3.5)
        text(x+3.0, y, label, ha='left', size=7.2)

    icon('results', 123, 30.3, 5.7)
    text(123, 24.3, 'Results', italic=True, size=8.5)
    for start, end in [(21.3, 25.9), (56.7, 61.3), (73, 76.8), (114.2, 118.4)]:
        arrow(start, 27.1, end, 27.1)

    # Lower anchors name checks actually performed, never established validity.
    anchors = [
        (6.0, 21.0, 16.5, 'audit', 'Feature\naudit'),
        (57.0, 21.0, 67.5, 'score', 'Brier\nscoring'),
        (112.5, 20.5, 123, 'comparison', 'Agent\ncomparisons'),
    ]
    for x, width, center, kind, label in anchors:
        ax.plot([center, center], [20.0, 10.3], color=black, lw=1.15,
                linestyle=(0, (3.5, 3.3)), zorder=2)
        container(x, 1.5, width, 8.8, blue, rounding=1.7)
        icon(kind, x+3.5, 5.9, 4.3)
        text(x+7.0, 5.9, label, ha='left', size=7.9, italic=True)

    arrow(27.2, 5.9, 56.8, 5.9, both=True)
    text(42, 8.8, 'Reused historical test', italic=True, size=7.7)
    arrow(78.2, 5.9, 112.3, 5.9, both=True)
    text(95.25, 8.8, 'Controlled comparisons', italic=True, size=7.7)

    visible_text = '\n'.join(labels)
    assert not re.search(r'\d', visible_text), 'The method figure must contain no numerals.'
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'method.{suffix}', dpi=280,
                    bbox_inches='tight', pad_inches=.035, facecolor='white')
    plt.close(fig)
    return visible_text


if __name__ == '__main__':
    draw()
