# build docker
docker build --tag spatial_covstat .

# launch script
docker run -it --rm --name my-running-script -v "$PWD":/usr/src/myapp -w /usr/src/myapp spatial_covstat python code.py