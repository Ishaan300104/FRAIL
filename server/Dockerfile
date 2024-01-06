# USE THE OFFICIAL PYTHON LANGUAGE AS THE BASE IMAGE 
FROM python:3.9-alpine

# SET THE WORKING DIRECTORY TO /app
WORKDIR /app

# Copy the Python file to the container
COPY ./inference.py .
COPY upload .

# Set the command to run when the container starts
CMD [ "python", "./inference.py" ]

