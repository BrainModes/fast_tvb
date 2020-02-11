FROM debian:latest
MAINTAINER Michael Schirner <michael.schirner@charite.de>

RUN apt-get update && \
    apt-get -y install \
    gcc \
    gsl-bin \
    libgsl0-dev 

COPY tvbii_multicore.c /tvbii_multicore.c
COPY compile_and_copy.sh /compile_and_copy.sh
RUN chmod 775 /compile_and_copy.sh
