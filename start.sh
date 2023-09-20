#/bin/bash
case "$1" in 
    "--slurm")
        srun \
            --job-name="sl2rl_job"  \
            --partition=gpu  \
            --time=5:00:00  \
            --ntasks=1  \
            --cpus-per-task=8  \
            --mem-per-cpu=4GB  \
            --pty \
            apptainer exec --writable-tmpfs --nv --bind $PWD image.sif /code tunnel --accept-server-license-terms 
    ;;
    "--run")
        apptainer exec --writable-tmpfs --nv--bind $PWD image.sif /code tunnel --accept-server-license-terms
    ;;
    "--build")
        apptainer build --build-arg WORKDIR=$PWD image.sif image.def
    ;;
    *)
    echo "Please provide one of the valid arguments: --slurm --run --build"
    exit 1
    ;;
esac