FROM python:3.6.6

# Update packages
RUN apt-get update

# Install prerequisite for matplotlib
RUN apt-get -y install libxft-dev libfreetype6 libfreetype6-dev

# Bundle app source
COPY . /src

EXPOSE 8000
WORKDIR /src

#install Python modules
RUN pip install -r requirements.txt

# Run it
ENTRYPOINT ["python", "application.py"]

# Environment Variables
ENV github_user your_github_id
ENV github_oauth_token your_github_read_only_token
ENV repo repo_name
