#!/usr/bin/env bash

head -n 50 patents_original.json >> patents0.json

head -n 45 patents_original.json >> patents1.json
tail -n 5 patents_original.json >> patents1.json

head -n 40 patents_original.json >> patents2.json
tail -n 10 patents_original.json >> patents2.json

head -n 35 patents_original.json >> patents3.json
tail -n 15 patents_original.json >> patents3.json

head -n 30 patents_original.json >> patents4.json
tail -n 20 patents_original.json >> patents4.json

head -n 25 patents_original.json >> patents5.json
tail -n 25 patents_original.json >> patents5.json

head -n 20 patents_original.json >> patents6.json
tail -n 30 patents_original.json >> patents6.json

head -n 15 patents_original.json >> patents7.json
tail -n 35 patents_original.json >> patents7.json

head -n 10 patents_original.json >> patents8.json
tail -n 40 patents_original.json >> patents8.json

head -n 5 patents_original.json >> patents9.json
tail -n 40 patents_original.json >> patents9.json

tail -n 50 patents_original.json >> patents10.json