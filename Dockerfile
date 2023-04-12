# Specify the base image
FROM python:3.9

# Install OS dependencies
RUN apt-get update

# Update pip
RUN pip install --upgrade pip ipython ipykernel
RUN ipython kernel install --name "python3" --user

# Set the working directory inside the container
WORKDIR /h2o2_ai

# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install required packages
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy all files and folders from dir
# Takes into account content in dockerignore
COPY . .

# Expose the necessary port(s)
EXPOSE 10101

ENTRYPOINT [ "wave", "run", "src.app.app.py"]