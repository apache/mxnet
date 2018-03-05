set -ex
apt install -y software-properties-common
add-apt-repository -y ppa:graphics-drivers
# Retrieve ppa:graphics-drivers and install nvidia-drivers.
# Note: DEBIAN_FRONTEND required to skip the interactive setup steps
apt update
DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends cuda-8-0