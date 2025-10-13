#!/bin/bash
set -e  # Exit on any error

echo "Starting setup"

echo "Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    gcc \
    g++ \
    make \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    libgl1-mesa-dev \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo
echo "Installing MuJoCo 2.1.0..."
mkdir -p ~/.mujoco
cd ~/.mujoco
wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm -f mujoco210-linux-x86_64.tar.gz

echo "Configuring MuJoCo dynamic linker..."
echo "$HOME/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco.conf
ldconfig

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Conda
echo "Creating conda environment from environment.yml..."
cd /workspace/offlineRLviaIO
conda env create -f environment.yml

echo "Precompiling mujoco_py..."
conda run -n io-agent python -c "import mujoco_py" || echo "mujoco_py precompilation attempted"

# Jupyter
echo "Setting up Jupyter kernel..."
conda run -n io-agent python -m ipykernel install --user --name io-agent --display-name "Python (io-agent)"
cat > ~/.local/share/jupyter/kernels/io-agent/kernel.json << 'EOF'
{
 "argv": [
  "/venv/io-agent/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (io-agent)",
 "language": "python",
 "env": {
  "LD_LIBRARY_PATH": "$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}",
  "MUJOCO_PY_MUJOCO_PATH": "$HOME/.mujoco/mujoco210"
 }
}
EOF

# Conda init
echo "Configuring conda activation in bashrc..."
if ! grep -q "conda.sh" ~/.bashrc; then
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
fi
if ! grep -q "conda activate io-agent" ~/.bashrc; then
    echo "conda activate io-agent" >> ~/.bashrc
fi

# MuJoCo init
echo "Adding MuJoCo environment variables to bashrc..."
if ! grep -q "MUJOCO_PY_MUJOCO_PATH" ~/.bashrc; then
    echo "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" >> ~/.bashrc
fi
if ! grep -q "\$HOME/.mujoco/mujoco210/bin" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\${LD_LIBRARY_PATH}" >> ~/.bashrc
fi

echo "Setup complete!"
