# Makefile
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

APP = main
APP_SRCS := $(wildcard src/*.c)

ifndef CORE
CORE=1
else
CORE= $(CORE)
endif


APP_CFLAGS += -DNUM_CORES=$(CORE) -Iinclude -O3 -w -fno-tree-loop-distribute-patterns -flto
APP_LDFLAGS += -lm -flto

include $(RULES_DIR)/pmsis_rules.mk
