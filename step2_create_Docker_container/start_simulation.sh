#!/bin/bash

FILE=/input
if [ ! -d "$FILE" ]; then
    echo "Error ($FILE): input directory does not exist. Terminating."
exit
fi

FILE=/output
if [ ! -d "$FILE" ]; then
    echo "Error ($FILE): output directory does not exist. Terminating."
exit
fi

/tvb $1 $2 $3
