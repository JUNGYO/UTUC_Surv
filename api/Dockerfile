# Use python 3.8 slim as base image
FROM python:3.8-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file to the container
COPY ./requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the start script
COPY start.sh /start.sh
# Give execute permissions to the start script
RUN chmod +x /start.sh

# Copy the rest of the application files to the container
COPY . .

# Use the start script as the default command
CMD ["/start.sh"]
