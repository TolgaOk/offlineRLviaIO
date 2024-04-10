#/bin/bash
case "$1" in 
    "--slurm")
        srun \
            --job-name="vscode-jupyter-server"  \
            --account=research-me-dcsc \
            --partition=compute \
            --time=9:00:00  \
            --ntasks=1  \
            --cpus-per-task=8  \
            --mem-per-cpu=2GB  \
            --gpus-per-task=0 \
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
