#!/bin/bash

base_dir="www.ic.unicamp.br/~helio/"
final_dir="imagens_pgm"
if [ -e "$final_dir" ];then
    echo $final_dir exists!
else
    wget -A .pgm -r -np http://${base_dir}/imagens_pgm/

    mv -r  $base_dir/imagens_pgm .
    rm -rf www.ic.unicamp.br
    echo files downloaded
fi
