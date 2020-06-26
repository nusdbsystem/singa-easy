FROM node:14.4.0-alpine3.11

WORKDIR /app

COPY web/package.json web/package.json
COPY web/yarn.lock web/yarn.loc

RUN cd web/ && yarn install

COPY web/ web/

EXPOSE 3002

CMD cd web/ && yarn start