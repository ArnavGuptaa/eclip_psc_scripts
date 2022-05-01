#!/bin/bash


#Change genome directory to /ocean/projects/mcb180074p/kghatole/eclip/STAR/genomehg38
#STAR RM REP

mkdir ./tempfiles/STAR
mkdir ./tempfiles/STAR/STARrmREP
STAR \
--runMode alignReads \
--runThreadN 64 \
--genomeDir ./STAR/genomeRM \
--genomeLoad NoSharedMemory \
--alignEndsType EndToEnd \
--outSAMunmapped Within \
--outFilterMultimapNmax 30 \
--outFilterMultimapScoreRange 1 \
--outFileNamePrefix ./tempfiles/STAR/STARrmREP/r1.trtr.sorted.STAR \
--outSAMtype BAM Unsorted \
--outFilterType BySJout \
--outBAMcompression 10 \
--outReadsUnmapped Fastx \
--outFilterScoreMin 10 \
--outSAMattrRGline ID:foo \
--outSAMattributes All \
--outSAMmode Full \
--outStd Log \
--readFilesIn ./tempfiles/cutadapt/r1.trtr.sorted.fq ./tempfiles/cutadapt/r2.trtr.sorted.fq


mv ./tempfiles/STAR/STARrmREP/r1.trtr.sorted.STARUnmapped.out.mate1 ./tempfiles/STAR/r1.STARrmREP.unmapped.fq
mv ./tempfiles/STAR/STARrmREP/r1.trtr.sorted.STARUnmapped.out.mate2 ./tempfiles/STAR/r2.STARrmREP.unmapped.fq

fastq-sort --id ./tempfiles/STAR/r1.STARrmREP.unmapped.fq > ./tempfiles/STAR/r1.STARrmREP.unmapped.sorted.fq
fastq-sort --id ./tempfiles/STAR/r2.STARrmREP.unmapped.fq > ./tempfiles/STAR/r2.STARrmREP.unmapped.sorted.fq

mkdir ./tempfiles/STAR/STARactual

#Change genome directory to /ocean/projects/mcb180074p/kghatole/eclip/STAR/genomehg38
STAR \
--runMode alignReads \
--runThreadN 64 \
--genomeDir ./STAR/genomehg38 \
--genomeLoad NoSharedMemory \
--readFilesIn \
./tempfiles/STAR/r1.STARrmREP.unmapped.sorted.fq \
./tempfiles/STAR/r2.STARrmREP.unmapped.sorted.fq \
--outSAMunmapped Within \
--outFilterMultimapNmax 1 \
--outFilterMultimapScoreRange 1 \
--outFileNamePrefix ./tempfiles/STAR/STARactual/r1.STAR \
--outSAMattributes All \
--outSAMtype BAM Unsorted \
--outFilterType BySJout \
--outReadsUnmapped Fastx \
--outFilterScoreMin 10 \
--outSAMattrRGline ID:foo \
--outStd Log \
--alignEndsType EndToEnd \
--outBAMcompression 10 \
--outSAMmode Full

mv ./tempfiles/STAR/STARactual/r1.STARAligned.out.bam ./tempfiles/STAR/r1.genome-mapped.bam
