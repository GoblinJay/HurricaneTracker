import subprocess
import sys
import platform

def run_command(command):
    """Run a command and print its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    
    return process.returncode

def main():
    # First, upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install core scientific packages using conda
    print("\nInstalling core packages with conda")
    conda_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pillow"
    ]
    
    for package in conda_packages:
        print(f"\nInstalling {package} with conda")
        result = run_command(f"conda install -y {package}")
        if result != 0:
            print(f"Error installing {package} with conda")
            sys.exit(1)
    
    # Install PyTorch CPU version
    print("\nInstalling PyTorch CPU version")
    result = run_command(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    if result != 0:
        print("Error installing PyTorch")
        sys.exit(1)
    
    # Install remaining packages with pip
    pip_packages = [
        "streamlit==1.0.0",
        "plotly==5.0.0",
        "folium==0.12.0",
        "streamlit-folium==0.11.0",
        "requests==2.26.0",
        "python-dateutil==2.8.0",
        "fastapi==0.68.0",
        "uvicorn==0.15.0",
        "python-multipart==0.0.5",
        "pydantic==1.8.0",
        "python-dotenv==0.19.0",
        "pytest==6.2.5",
        "black==21.7b0",
        "flake8==3.9.0",
        "netCDF4==1.5.7",
        "xarray==0.19.0",
        "cartopy==0.19.0"
    ]
    
    for package in pip_packages:
        print(f"\nInstalling {package}")
        result = run_command(f"{sys.executable} -m pip install {package}")
        
        if result != 0:
            print(f"Error installing {package}")
            sys.exit(1)
    
    print("\nAll packages installed successfully!")
    
    # Verify PyTorch installation
    try:
        import torch
        print("\nPyTorch Status:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"Operating System: {platform.system()} {platform.release()}")
        print("Note: Running on CPU mode for Windows with AMD GPU")
    except ImportError:
        print("\nError: PyTorch not installed correctly")

if __name__ == "__main__":
    main() 