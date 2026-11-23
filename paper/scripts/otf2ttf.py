"""Convert a CFF-flavoured OpenType font (.otf) into a TrueType one (.ttf), so matplotlib can embed it in PDFs as a
regular Type 42 font. Inter and Source Serif 4 ship as .otf in TeX Live; matplotlib embeds those as an OpenType
font file under a TrueType font type, which some PDF readers flag ("Mismatch between font type and embedded font
file"). Based on the otf2ttf snippet that comes with fontTools (cubic outlines to quadratic with cu2qu).

    python3 scripts/otf2ttf.py IN.otf OUT.ttf
"""
import sys

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

MAX_ERR = 1.0  # font units


def otf_to_ttf(src, dst):
    font = TTFont(src)
    assert font.sfntVersion == "OTTO" and "CFF " in font, f"{src} is not a CFF OpenType font"
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in glyph_set.keys():
        tt_pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(tt_pen, MAX_ERR, reverse_direction=True))
        glyphs[name] = tt_pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():
        if hasattr(glyph, "xMin"):
            hmtx[name] = (hmtx[name][0], glyph.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max((len(getattr(g, "components", [])) for g in glyf.glyphs.values()), default=0)
    maxp.compile(font)
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font["head"].glyphDataFormat = 0
    font.sfntVersion = "\000\001\000\000"
    font.save(dst)


if __name__ == "__main__":
    otf_to_ttf(sys.argv[1], sys.argv[2])
