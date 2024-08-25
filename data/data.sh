# Setup data directory
mkdir -p data/raw/exist
mkdir -p data/raw/csmb
mkdir -p data/preprocessed/exist
mkdir -p data/preprocessed/csmb
mkdir -p data/augmented/exist
mkdir -p data/augmented/csmb

# Download the data files and put it in respective folders (raw and preprocessed). The augmented dataset can be generated
# using src/stages/augment_data.py
