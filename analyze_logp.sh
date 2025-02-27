#!/bin/bash
for prompt in "a turtle with a bow" "a black car and a white clock" "a frog and a mouse"; do
    for same_noise in True False; do
        python -m analyze_logp --prompt="$prompt" --same_noise=$same_noise
    done
done
