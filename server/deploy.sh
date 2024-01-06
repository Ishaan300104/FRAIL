#!/bin/bash

# 1. Build the project
docker build -t flask-server:1.0 .
docker run -d flask-server:1.0
