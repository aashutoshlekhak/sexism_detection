# Setup data directory
mkdir -p data/raw/exist
mkdir -p data/preprocessed/exist
mkdir -p data/preprocessed/csmb
mkdir -p data/augmented/exist
mkdir -p data/raw/csmb
cd data/raw/exist
echo "Downloading exist dataset"
gdown https://drive.google.com/uc?id=1-5L3Q_qy6lpOQUDrNM9q_4q0wwGR-zFr
gdown https://drive.google.com/uc?id=1-6jjL9krpxh680Z2fFfEsbl81EvXbK_o

cd ../../.. 
cd data/raw/csmb
echo "Downloading csmb dataset"
gdown https://drive.google.com/uc?id=13bxeva7zLSDmWqsSU3EqY6QdFNR03Bca

cd ../../..
cd data/augmented/exist
echo "Downloading augmented exist dataset"
gdown https://drive.google.com/uc?id=13OOidQupiz5P2upmYfp40hKybpgo7SeC


