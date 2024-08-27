# Setup data directory
mkdir -p data/raw/exist
cd data/raw/exist
echo "Downloading exist dataset"
gdown https://drive.google.com/uc?id=1-5L3Q_qy6lpOQUDrNM9q_4q0wwGR-zFr
gdown https://drive.google.com/uc?id=1-C5K9x2-rQAAi7wRwt4aMqkW8qcntLqC

cd ../../.. 
mkdir -p data/raw/csmb
cd data/raw/csmb
echo "Downloading csmb dataset"
gdown https://drive.google.com/uc?id=1jfE609X87uqpUWMLiJq_HN7Y7BVLlO_c
mkdir -p data/preprocessed/exist
mkdir -p data/preprocessed/csmb
mkdir -p data/augmented/exist
mkdir -p data/augmented/csmb

