docker stop seweb
docker rm seweb
echo "Docker container is removed"

docker image rm seweb --force
echo "Docker image is deleted"