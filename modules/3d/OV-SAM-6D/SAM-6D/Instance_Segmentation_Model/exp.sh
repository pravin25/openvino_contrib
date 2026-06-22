#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# with sam
python run_inference.py dataset_name=icbin

# with fastsam
python run_inference.py dataset_name=icbin model=ISM_fastsam
