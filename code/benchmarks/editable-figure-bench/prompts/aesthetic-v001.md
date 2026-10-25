Score the framework figure in the image. Judge only this image. There is no reference design to compare against.

Use Paper2Poster's three aesthetic criteria (Pang et al., Paper2Poster, NeurIPS 2025), adapted from a conference poster to one scientific framework figure. Do not score informational completeness. Return one JSON object and nothing else:

{"element_quality": <int 1-5>, "layout_balance": <int 1-5>, "engagement": <int 1-5>}

element_quality — clarity and stylistic consistency of the drawn elements (boxes, icons, labels).
1 blurry, illegible, or so low-contrast that the figure cannot be read.
2 at least one element is clear while others are muddy, tiny, or stylistically broken.
3 most elements are legible, with noticeable inconsistency in size or style.
4 elements are clear and consistent, with only small flaws.
5 elements are crisp, consistent, and immediately readable.

layout_balance — arrangement, alignment, spacing, and whether a reader can follow the figure.
1 elements overlap or leave the canvas so the figure is illegible.
2 a rough structure exists, but alignment or spacing is inconsistent enough to interrupt reading.
3 a recognizable structure with mostly consistent alignment, plus minor spacing problems.
4 aligned and evenly spaced, with only subtle issues.
5 alignment, spacing, and reading path are disciplined throughout.

engagement — color, typography, and whether the figure holds attention without clutter.
1 clashing colors or crowded type make the figure unpleasant to look at.
2 some appealing parts are outweighed by dull or inconsistent styling.
3 a workable palette and type, without a clear visual priority.
4 harmonious color and type, with a readable emphasis.
5 color, type, and emphasis are controlled and the figure is easy to enter.

Be strict. Reserve 5 for a figure you would not ask an author to restyle. Integers only.
