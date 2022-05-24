sudo apt update
sudo apt upgrade -y

sudo apt install -y zip
curl -s "https://get.sdkman.io" | bash
source .bashrc
source "${HOME}/.sdkman/bin/sdkman-init.sh"

sdk install java 11.0.11-open
sdk install sbt

git clone https://github.com/albertqjiang/Portal-to-ISAbelle.git
cd Portal-to-ISAbelle
sbt compile

cd ~
wget https://isabelle.in.tum.de/website-Isabelle2021/dist/Isabelle2021_linux.tar.gz
tar -xzf Isabelle2021_linux.tar.gz

# This step takes ~6 hours
# ~/Isabelle2021/bin/isabelle build -b -D Isabelle2021/src/HOL

# Those steps take ~24 hours
wget https://www.isa-afp.org/release/afp-2021-10-22.tar.gz
tar -xzf afp-2021-10-22.tar.gz
# export AFP=afp-2021-10-22/thys
# ~/Isabelle2021/bin/isabelle build -b -D $AFP
wget https://storage.googleapis.com/n2formal-public-data/isabelle_heaps.tar.gz
tar -xzf isabelle_heaps.tar.gz

cd Portal-to-ISAbelle
tar -xzf universal_test_theorems.tar.gz
