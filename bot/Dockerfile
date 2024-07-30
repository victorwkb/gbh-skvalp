FROM python:3.11.6

# Copy just what's required to do a `pip install`, because that's what's going to take time.
# Hence, docker will cache this step and the next step.
# If we'd copied the whole directory here, __any__ change in that directory would bust the cache
# and force docker to do `pip install` again.
# Install COIN-OR CBC solver
RUN apt-get update && apt-get install -y coinor-cbc coinor-libcbc-dev

RUN apt-get install -y libhdf5-dev libhdf5-serial-dev

COPY requirements.txt /bot/requirements.txt

RUN pip install --no-cache-dir -r bot/requirements.txt

COPY . /bot

WORKDIR /



