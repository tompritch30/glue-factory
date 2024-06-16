#!/bin/bash

# Define the SSH key and log file path
SSH_KEY="/homes/tp4618/.ssh/id_rsa"
LOG_FILE="/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/syntheticForestData/gpu_memory_usage.log"
MEMORY_THRESHOLD_MB=5000  # Set threshold for available memory (in MiB)

# Start logging
echo "Starting GPU memory check at $(date)" | tee "${LOG_FILE}"

# Initialize an empty array to hold machines with more than specified MBs of GPU memory available
high_memory_machines=()

# Loop through the range of GPUs
for i in {1..30}; do
    machine="gpu$(printf "%02d" $i)"
    echo "Checking GPU memory on ${machine}..."

    # Run nvidia-smi command remotely to fetch the memory usage
    # Parsing the output to find lines that include "MiB /", then using awk to get the used and total memory values
    # Fetch the used and total memory values separately
    # Fetch the used and total memory values separately
    memory_used=$(ssh -i "${SSH_KEY}" "${machine}" nvidia-smi |
                awk -F'|' '/MiB \// {print $3}' | awk '{print $1}' | sed 's/MiB//')
    memory_total=$(ssh -i "${SSH_KEY}" "${machine}" nvidia-smi |
                awk -F'|' '/MiB \// {print $3}' | awk '{print $3}' | sed 's/MiB//')


    # Calculate available memory
    memory_available=$((memory_total - memory_used))

    # Check if the available memory is greater than the threshold
    if [ "$memory_available" -ge "$MEMORY_THRESHOLD_MB" ]; then
        echo "${machine} has ${memory_available} MiB of GPU memory available." | tee -a "${LOG_FILE}"
        high_memory_s=$(($memory_total - $memory_used))
        high_memory_machines+=("${machine}:${memory_available}MiB")
    else
        echo "${machine} is using ${memory_used} MiB out of ${memory_total} available." | tee -a "${LOG_FILE}"
    fi
done

# Print machines with high available memory
echo "Machines with ${MEMORY_THRESHOLD_MB} MiB or more of GPU memory available:" | tee -a "${LOG_FILE}"
printf "%s\n" "${high_memory_machines[@]}" | tee -a "${LOG_FILE}"

echo "GPU memory check completed at $(date)" | tee -a "${LOG_FILE}"
