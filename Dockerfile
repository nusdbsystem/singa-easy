FROM node:14.7.0-alpine3.12

WORKDIR /app

COPY web/package.json web/package.json
COPY web/yarn.lock web/yarn.loc

RUN cd web/ && yarn install

COPY web/ web/

EXPOSE 3002

CMD cd web/ && yarn start