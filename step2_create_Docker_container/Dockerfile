FROM debian:latest
MAINTAINER Michael Schirner <michael.schirner@charite.de>

RUN apt-get update && \
    apt-get -y install \
    gcc \
    gsl-bin \
    libgsl0-dev 

COPY tvb /tvb
COPY start_simulation.sh /start_simulation.sh
RUN chmod 775 /start_simulation.sh
