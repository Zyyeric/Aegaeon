#! /bin/bash
set -ex

export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113
export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export NO_PROXY=127.0.0.1,0.0.0.0

apt update
apt install -y openssh-server
echo -e "PermitRootLogin yes\nPasswordAuthentication yes\nPort 2222" | sudo tee -a /etc/ssh/sshd_config > /dev/null
sudo service ssh restart
echo "root:123456wyt" | chpasswd
