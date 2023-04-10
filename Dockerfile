# Specify the base image
FROM python:3.9

# Install OS dependencies
RUN apt-get update

# Update pip
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /h2o2_ai

# Copy all files and folders from dir
# Takes into account content in dockerignore
COPY . .

# Install required packages
RUN pip3 install -r requirements.txt

# Expose the necessary port(s)
EXPOSE 10101

ENTRYPOINT [ "wave", "run", "src.app.app.py"]
