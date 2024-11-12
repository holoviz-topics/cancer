#!/usr/bin/python
"""Convert a .hic file into HoloSeq hseq format. Data format decribed here:
https://github.com/fubar2/holoSeq/blob/main/HoloSeqOverview.md
"""

import gzip
import sys
import logging

import hicstraw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GzipOut:
    def __init__(self, fn):
        self.fn = fn
        self.f = gzip.open(fn, "wb")

    def write(self, x):
        self.f.write(str.encode(x))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
    

def convert_hic_to_hseq(hicfn, ostream, max_chrom, title):
    hic = hicstraw.HiCFile(hicfn)
    chroms = hic.getChromosomes()
    if max_chrom < 1:
        max_chrom = len(chroms)
    resolution = min(hic.getResolutions())

    ## write header
    ostream.write(
        f"@v1HoloSeq2D\n@title {title}\n" + 
        "".join([f"@H1 {c.name} {c.length}\n" for c in chroms])
    )
    
    xbase = 0
    for i, cx in enumerate(chroms[:max_chrom]):
        ybase = 0
        for cy in chroms[0:(i + 1)]:
            logging.info(f"Starting on {cx.name} vs. {cy.name}")

            ## I'm not sure what BP/FRAG means at the moment...
            #method = "BP" if cx.name == cy.name else "FRAG"
            method = "BP"

            ## This was an older API
            # xcoords, ycoords, counts = hicstraw.straw('VC', hicfn, cx.name, cy.name, method, resolution)
            # for xcoord, ycoord, count in zip(xcoords, ycoords, counts):
            #     ostream.write(f"{xbase + xcoord} {ybase + ycoord} {count}\n")

            for result in hicstraw.straw("observed", 'NONE', hicfn, cx.name, cy.name, method, resolution):
                ostream.write(f"{xbase + result.binX} {ybase + result.binY}\n" * int(result.counts))
            ybase += cy.length
        xbase += cx.length


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", default="", help="matrix title (default: input hic path)")
    ap.add_argument("--max-chrom", type=int, default=0, help="maximum chromosomes to convert")
    ap.add_argument("hicfile", help="hic format filename or URL")
    ap.add_argument("hseqgz", help="output filename (will be gzipped)")
    argv = ap.parse_args()

    argv.title = argv.title or argv.hicfile
    with GzipOut(argv.hseqgz) as ostream:
        convert_hic_to_hseq(argv.hicfile, ostream, argv.max_chrom, argv.title)
