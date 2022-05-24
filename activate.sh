#!/bin/bash
# on a node with GPU, e.g., asimov-8, run source activate.sh
root_dir=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
curr_dir=$(pwd)
if [[ ! -d $root_dir/TC-venv ]]
then
  echo "Installing transformer distillation virtual environment ..."
  python -m venv $root_dir/TD-venv
  need_install=true
else
  need_install=false
fi

source $root_dir/TC-venv/bin/activate

if $need_install
then
  cd $root_dir
  pip install --upgrade pip
  pip install -r requirements.txt

  # amp support
  export PATH=${YOUR_PATH_TO_GCC_VERSION7_5_0}/bin:${PATH}
  alias gcc='${YOUR_PATH_TO_GCC_VERSION7_5_0}/bin/gcc'
  alias g++='${YOUR_PATH_TO_GCC_VERSION7_5_0}/bin/g++'
  git clone https://github.com/NVIDIA/apex $root_dir/TC-venv/apex
  cd $root_dir/TC-venv/apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
  cd $curr_dir
fi
