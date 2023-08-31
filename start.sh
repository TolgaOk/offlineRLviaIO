#/bin/bash
case "$1" in 
    "--slurm")
        srun 
            --job-name="sl2rl_job"  \
            --partition=compute  \
            --time=01:00:00  \
            --ntasks=1  \
            --cpus-per-task=4  \
            --mem-per-cpu=2GB  \
            --pty \
            apptainer exec --bind $PWD image.sif /code tunnel --accept-server-license-terms 
    ;;
    "--run")
        apptainer exec --bind $PWD image.sif /code tunnel --accept-server-license-terms
    ;;
    "--build")
        apptainer build --build-arg WORKDIR=$PWD image.sif image.def
    ;;
    *)
    echo "Please provide one of the valid arguments: --slurm --run --build"
    exit 1
    ;;
esac