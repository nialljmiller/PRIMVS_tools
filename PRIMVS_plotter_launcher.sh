#!/bin/bash

for i in {0..7}
do
    python PRIMVS_plotter.py $i &
done


