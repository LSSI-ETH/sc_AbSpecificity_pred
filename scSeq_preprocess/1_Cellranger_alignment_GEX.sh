#!/bin/bash

# Display help message
show_help() {
    echo "Usage: $0 -cellrangerpath /path/to/cellranger -fastqfilepath /path/to/fastqfiles -ref_path /path/to/reference -outpath /path/to/output"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -cellrangerpath) cellrangerpath="$2"; shift ;;
        -fastqfilepath) fastqfilepath="$2"; shift ;;
        -ref_path) ref_path="$2"; shift ;;
        -outpath) outpath="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Check if all required arguments are provided
if [ -z "$cellrangerpath" ] || [ -z "$fastqfilepath" ] || [ -z "$ref_path" ] || [ -z "$outpath" ]; then
    echo "Error: All arguments -cellrangerpath, -fastqfilepath, -ref_path, and -outpath are required."
    show_help
    exit 1
fi

cd "outpath"
"cellrangerpath" count --id=OVA_pos1 --transcriptome="ref_path" --fastqs="$fastqfilepath/OVA_pos1" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
"cellrangerpath" count --id=OVA_neg1 --transcriptome="ref_path" --fastqs="$fastqfilepath/OVA_neg1" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
echo "OVA1 done"

"cellrangerpath" count --id=OVA_pos2 --transcriptome="ref_path" --fastqs="$fastqfilepath/OVA_pos2" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
"cellrangerpath" count --id=OVA_neg2 --transcriptome="ref_path" --fastqs="$fastqfilepath/OVA_neg2" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
echo "OVA2 done"

"cellrangerpath" count --id=RBD_pos --transcriptome="ref_path" --fastqs="$fastqfilepath/RBD_pos" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
"cellrangerpath" count --id=RBD_neg --transcriptome="ref_path" --fastqs="$fastqfilepath/RBD_neg" --no-bam --localmem=20 --include-introns=true --chemistry=fiveprime
echo "RBD done"

