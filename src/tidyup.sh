#!/usr/bin/env bash

find $1 -type f -name "padded_*" -delete
find $1 -type f -name "coregd_*" -delete
