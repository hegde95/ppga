#!/bin/bash
grep --color -r --exclude-dir=wandb --exclude-dir=env $@
