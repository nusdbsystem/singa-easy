docker build -t seweb .
echo "Docker image for Singa-Easy application is built"

docker run -it --publish 3002:3000 --name seweb seweb:latest
echo "Docker container is built and the application is deployed on port 3002"