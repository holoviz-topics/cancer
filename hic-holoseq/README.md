# HoloSeq HiC Data

Steps to getting HTAN data in HoloSeq:

1. create and activate `environment.yml`
2. download a .hic file; Go to
   [GSE207951](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207951),
   click on "(custom)" in the table at the bottom, check a single .hic
   file, then click the "Download" button. Untar it.
3. Run `python3 hic2hseq.py --max-chrom 3
   GSM6326543_A001C007.hg38.nodups.pairs.hic test.hseq.gz` to create a
   small hseq.gz file (29MB with that sample).
4. checkout [HoloSeq](https://github.com/fubar2/holoSeq/tree/main),
   create its venv, and activate that.
5. run `panel serve holoseq_display.py --show --args --inFile ../test.hseq.gz --size 1000`


An pre-converted HTAN file for [GSM6326543 is available(530MB)](https://pub-867b121072f54b4a9eecdf01cd27246b.r2.dev/GSM6326543_A001C007.hg38.nodups.pairs.hseq.gz)
