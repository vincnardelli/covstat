# build docker
docker build --tag covstat_model .

# launch script
docker run --rm --name my-running-script -v "$PWD":/usr/src/myapp -w /usr/src/myapp covstat_model python model.py