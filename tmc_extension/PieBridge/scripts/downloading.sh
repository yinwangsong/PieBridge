echo "downloading proxy subnetworks..."

gdown 12VrLe3Nf7NLx7VjReMXOoqCjIWiiZoj- -O ./proxynetworks.tar.gz

echo "downloading datasets..."

gdown 15iwkPHR1RutYDz4Yq4DGByYV78o51Z7e -O ./datasets.tar.gz

echo "removing existing folders..."

rm -r ./datasets
rm -r ./proxynetworks

echo "decompressing..."

tar -xzvf ./proxynetworks.tar.gz
tar -xzvf ./datasets.tar.gz