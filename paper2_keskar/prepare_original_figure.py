"""Fetch and crop the *actual published* Figure 3(a,c,d), not digitize/redraw it.
Run on a computer with internet access:
  python -m pip install pymupdf pillow
  python prepare_original_figure.py
Requires arxiv.org access. The report compiles even before this script is run.
"""
from pathlib import Path
import io
from urllib.request import Request, urlopen
import fitz
from PIL import Image, ImageOps

PDF_URL = "https://arxiv.org/pdf/1609.04836"
TARGET = Path(__file__).resolve().parent / "figures" / "orig_fig3_F1_C1_C2.png"

def render_panel(page, bounds, zoom=3.0):
    """bounds: page-relative (x0,y0,x1,y1) enclosing entire panel & label."""
    r = page.rect
    x0, y0, x1, y1 = bounds
    clip = fitz.Rect(r.x0+x0*r.width, r.y0+y0*r.height,
                     r.x0+x1*r.width, r.y0+y1*r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom), clip=clip, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def main():
    req = Request(PDF_URL, headers={"User-Agent":"Mozilla/5.0 (academic figure reproduction)"})
    with urlopen(req, timeout=60) as r:
        pdfbytes = r.read()
    doc=fitz.open(stream=pdfbytes,filetype="pdf")
    # ICLR 2017 arXiv v2: Figure 3 sits on printed page 6, PDF page index 5.
    page=doc[5]
    panels = [
        render_panel(page,(.167,.102,.500,.300)), # original (a): F1
        render_panel(page,(.167,.303,.500,.500)), # original (c): C1
        render_panel(page,(.503,.303,.847,.500)), # original (d): C2
    ]
    desired_h=max(im.height for im in panels)
    panels=[im.resize((round(im.width*desired_h/im.height),desired_h)) for im in panels]
    gap=16; margin=20
    total_w=sum(im.width for im in panels)+gap*(len(panels)-1)+2*margin
    out=Image.new('RGB',(total_w,desired_h+2*margin),'white')
    x=margin
    for im in panels:
        out.paste(im,(x,margin));x+=im.width+gap
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    out.save(TARGET,optimize=True)
    print(f'Saved authentic, cropped original panels to {TARGET}')
    print('Inspect the crop visually. If the arXiv PDF layout changes, adjust panel bounds.')

if __name__=='__main__':
    main()
