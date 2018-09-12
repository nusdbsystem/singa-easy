IMAGE_NAME=rafiki_model

usage()  {
  echo "Usage: $0 <service_name> <service_id>"
  exit 1
}

if [ $# -ne 3 ] ; then
    usage
fi

docker service create --name $1 \
  --rm --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e ADMIN_HOST=$ADMIN_HOST \
  -e ADMIN_PORT=$ADMIN_PORT \
  -e SUPERADMIN_EMAIL=$SUPERADMIN_EMAIL \
  -e SUPERADMIN_PASSWORD=$SUPERADMIN_PASSWORD \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -v $LOGS_FOLDER_PATH:$LOGS_FOLDER_PATH \
  $IMAGE_NAME $2
