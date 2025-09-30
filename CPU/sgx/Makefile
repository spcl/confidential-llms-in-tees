# SPDX-License-Identifier: LGPL-3.0-or-later

# LLM and the pre-trained model must be installed on the system. See README for details.

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = none
endif

#.PHONY: all
#all: LLM.manifest
#ifeq ($(SGX),1)
all: LLM.manifest.sgx LLM.sig
#endif

LLM.manifest: llm.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(realpath $(shell sh -c "command -v /home/ubuntu/miniconda3/envs/py310/bin/python3")) \
		$< > $@
# /home/sdp/intel-extension-for-pytorch/examples/cpu/inference/python/llm/runLLamaInference.sh
# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# for details on this workaround see
# https://github.com/gramineproject/gramine/blob/e8735ea06c/CI-Examples/helloworld/Makefile
LLM.manifest.sgx LLM.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: LLM.manifest
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

.PHONY: run
run: LLM.manifest.sgx LLM.sig
	FI_PROVIDER=psm3 gramine-sgx LLM ~/intel-extension-for-pytorch/examples/cpu/inference/python/llm/deepspeed_generate.py >out 2>&1

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) *.pt result.txt
