# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install OS dependencies (including LaTeX for report generation)
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make run.sh executable
RUN chmod +x run.sh

# Default command
CMD ["./run.sh"]
