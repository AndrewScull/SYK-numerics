# Use make as a basic job scheduler to run calculations.
#
# Make can already manage a set of jobs with a limit on the number running
# simultaneously and will avoid re-running jobs that have already produced
# their results. That's good enough to avoid having to come up with a custom
# job scheduler.

dt := 0.000001
qs := $(shell seq 2 2 10)
temps := $(shell seq -2.6 0.1 -1.9)

# Returns the name of the results file given a set of inputs for the calculation.
#
# $(1) - q
# $(2) - temp
define result_file
	dt_$(dt)_q_$(1)_temp_$(2).result
endef

# Increase threads to speed up the FFT a bit if there are unused cores. It
# doesn't speed up linearly with the number of threads to it's better to run
# more simultaneous jobs on the available cores.
PYFFTW_NUM_THREADS ?= 1

# Generate rule to run the calculation with the given parameters and store the
# result in a file ready for post processing.
#
# $(1) - q
# $(2) - temp
define calculation
$$(call result_file,$(1),$(2)):
	PYFFTW_NUM_THREADS=$(PYFFTW_NUM_THREADS) time python ../large_N_entropy_from_SD.py --timestep=$(dt) --fermions=$(1) --temperature=$(2) --output=$$@ > $$@.log 2>&1
endef

# List all the combinations of results based on the inputs.
results = $(foreach q,$(qs),$(foreach temp,$(temps),$(call result_file,$(q),$(temp))))

# Rule to run all calculations for their results.
all: $(results)

# Create rules for each calculation.
$(foreach q,$(qs),$(foreach temp,$(temps),$(eval $(call calculation,$(q),$(temp)))))
