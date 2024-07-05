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

"cellrangerpath" vdj --id=OVA_pos1  --reference="ref_path" --fastqs="$fastqfilepath/OVA_pos1" --sample=OVA_pos1 --localmem=20
"cellrangerpath" vdj --id=OVA_neg1  --reference="ref_path" --fastqs="$fastqfilepath/OVA_neg1" --sample=OVA_neg1 --localmem=20
echo "OVA1 done"

"cellrangerpath" vdj --id=OVA_pos2  --reference="ref_path" --fastqs="$fastqfilepath/OVA_pos2" --sample=OVA_pos2 --localmem=20
"cellrangerpath" vdj --id=OVA_neg2  --reference="ref_path" --fastqs="$fastqfilepath/OVA_neg2" --sample=OVA_neg2 --localmem=20
echo "OVA2 done"

"cellrangerpath" vdj --id=RBD_pos  --reference="ref_path" --fastqs="$fastqfilepath/RBD_pos" --sample=RBD_pos --localmem=20
"cellrangerpath" vdj --id=RBD_neg  --reference="ref_path" --fastqs="$fastqfilepath/RBD_neg" --sample=RBD_neg --localmem=20
echo "RBD done"

# cellranger vdj --id=HEL_negative --reference=/refdata-cellranger-vdj-GRCm38-alts-ensembl-7.0.0/ --fastqs=./for_cellranger/HEL_negative_VDJ/ --sample=VDJ_HEL_negative --localmem=20
bsub -W 2880 -R 'rusage[mem=20000]' /cluster/scratch/lerlach/scSeq/cellranger-7.1.0/cellranger vdj --id=HA_positive --reference=/cluster/scratch/lerlach/scSeq/refdata-cellranger-vdj-GRCm38-alts-ensembl-7.0.0/ --fastqs=/cluster/scratch/lerlach/scSeq/HEL_OMI_HA_seq/for_cellranger/HA_positive_VDJ/ --sample=VDJ_HA_positive --localmem=20
bsub -W 2880 -R 'rusage[mem=20000]' /cluster/scratch/lerlach/scSeq/cellranger-7.1.0/cellranger vdj --id=HA_negative --reference=/cluster/scratch/lerlach/scSeq/refdata-cellranger-vdj-GRCm38-alts-ensembl-7.0.0/ --fastqs=/cluster/scratch/lerlach/scSeq/HEL_OMI_HA_seq/for_cellranger/HA_negative_VDJ/ --sample=VDJ_HA_negative --localmem=20
bsub -W 2880 -R 'rusage[mem=20000]' /cluster/scratch/lerlach/scSeq/cellranger-7.1.0/cellranger vdj --id=RBD_positive --reference=/cluster/scratch/lerlach/scSeq/refdata-cellranger-vdj-GRCm38-alts-ensembl-7.0.0/ --fastqs=/cluster/scratch/lerlach/scSeq/HEL_OMI_HA_seq/for_cellranger/RBD_positive_VDJ/ --sample=VDJ_RBD_positive --localmem=20
bsub -W 2880 -R 'rusage[mem=20000]' /cluster/scratch/lerlach/scSeq/cellranger-7.1.0/cellranger vdj --id=RBD_negative --reference=/cluster/scratch/lerlach/scSeq/refdata-cellranger-vdj-GRCm38-alts-ensembl-7.0.0/ --fastqs=/cluster/scratch/lerlach/scSeq/HEL_OMI_HA_seq/for_cellranger/RBD_negative_VDJ/ --sample=VDJ_RBD_negative --localmem=20
