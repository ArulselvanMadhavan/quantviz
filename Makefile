CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)
MODELS_DIR ?= $(shell pwd)
PORT ?= 8080
OCAML_COMPILER := 4.14.1

DC_RUN_VARS := USER_NAME=${USER} \
	CURRENT_UID=${CURRENT_UID} \
	CURRENT_GID=${CURRENT_GID} \
	OCAML_COMPILER=${OCAML_COMPILER} \
	MODELS_DIR=${MODELS_DIR} \
	PORT=${PORT}

all:
	@dune build @all

format:
	@dune build @fmt --auto-promote

WATCH ?= @all
watch:
	@dune build $(WATCH) -w

clean:
	@dune clean

.PHONY: quantviz
quantviz:
	sudo ${DC_RUN_VARS} docker compose -f docker-compose.yml run --service-ports quantviz bash

.PHONY: diffusers-ocaml-rebuild
quantviz-rebuild:
	sudo ${DC_RUN_VARS} docker compose -f docker-compose.yml build

.PHONY: kill
kill:
	sudo docker kill $(shell docker ps -q)
