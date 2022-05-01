#!/bin/bash

# fread=$1
# rread=$2




# conda activate eclip

# umi_tools extract \
# --extract-method=string \
# --bc-pattern=NNNNNNNNNN \
# --bc-pattern2=NNNNNNNNNN \
# --log ./tempfiles/EXAMPLE_SE.rep1_clip.metrics \
# --read2-in ./DataSet/ENCFF840EOA.fastq.gz \
# --read2-out ./tempfiles/EXAMPLE_SE.rep1.umi.r2.fq \
# --stdin ./DataSet/ENCFF950SML.fastq.gz \
# --stdout ./tempfiles/EXAMPLE_SE.rep1.umi.r1.fq

# conda deactivate

mkdir ./tempfiles/cutadapt

#CutAdapt round 1
cutadapt \
--match-read-wildcards \
--times 1 \
-e 0.1 \
-O 1 \
--quality-cutoff 6 \
-m 18 \
-j 64 \
-a NNNNNAGATCGGAAGAGCACACGTCTGAACTCCAGTCAC \
-g CTTCCGATCTACAAGTT \
-g CTTCCGATCTTGGTCCT \
-A AACTTGTAGATCGGA \
-A AGGACCAAGATCGGA \
-A ACTTGTAGATCGGAA \
-A GGACCAAGATCGGAA \
-A CTTGTAGATCGGAAG \
-A GACCAAGATCGGAAG \
-A TTGTAGATCGGAAGA \
-A ACCAAGATCGGAAGA \
-A TGTAGATCGGAAGAG \
-A CCAAGATCGGAAGAG \
-A GTAGATCGGAAGAGC \
-A CAAGATCGGAAGAGC \
-A TAGATCGGAAGAGCG \
-A AAGATCGGAAGAGCG \
-A AGATCGGAAGAGCGT \
-A GATCGGAAGAGCGTC \
-A ATCGGAAGAGCGTCG \
-A TCGGAAGAGCGTCGT \
-A CGGAAGAGCGTCGTG \
-A GGAAGAGCGTCGTGT \
-o ./tempfiles/cutadapt/r1.tr.fq \
-p ./tempfiles/cutadapt/r2.tr.fq \
./tempfiles/EXAMPLE_SE.rep1.umi.r1.fq \
./tempfiles/EXAMPLE_SE.rep1.umi.r2.fq

cutadapt \
--match-read-wildcards \
--times 1 \
-e 0.1 \
-O 1 \
--quality-cutoff 6 \
-m 18 \
-j 64 \
-A AACTTGTAGATCGGA \
-A AGGACCAAGATCGGA \
-A ACTTGTAGATCGGAA \
-A GGACCAAGATCGGAA \
-A CTTGTAGATCGGAAG \
-A GACCAAGATCGGAAG \
-A TTGTAGATCGGAAGA \
-A ACCAAGATCGGAAGA \
-A TGTAGATCGGAAGAG \
-A CCAAGATCGGAAGAG \
-A GTAGATCGGAAGAGC \
-A CAAGATCGGAAGAGC \
-A TAGATCGGAAGAGCG \
-A AAGATCGGAAGAGCG \
-A AGATCGGAAGAGCGT \
-A GATCGGAAGAGCGTC \
-A ATCGGAAGAGCGTCG \
-A TCGGAAGAGCGTCGT \
-A CGGAAGAGCGTCGTG \
-A GGAAGAGCGTCGTGT \
-o ./tempfiles/cutadapt/r1.trtr.fq \
-p ./tempfiles/cutadapt/r2.trtr.fq \
./tempfiles/cutadapt/r1.tr.fq \
./tempfiles/cutadapt/r2.tr.fq

# fastqc -o ./output/fastqc2/read1 -t 8 ./tempfiles/cutadapt/r1.trtr.fq
# fastqc -o ./output/fastqc2/read2 -t 8 ./tempfiles/cutadapt/r2.trtr.fq
fastq-sort --id ./tempfiles/cutadapt/r1.trtr.fq > ./tempfiles/cutadapt/r1.trtr.sorted.fq
fastq-sort --id ./tempfiles/cutadapt/r2.trtr.fq > ./tempfiles/cutadapt/r2.trtr.sorted.fq

