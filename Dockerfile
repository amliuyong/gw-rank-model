ARG REGISTRY_URI
FROM ${REGISTRY_URI}/tensorflow-training:1.15.5-gpu-py37-cu100-ubuntu18.04


RUN apt-get update
RUN apt-get install -y --no-install-recommends nginx curl \
&& pip3 install --upgrade pip
RUN apt-get install -y --no-install-recommends python3.7


RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends ca-certificates
RUN apt-get install -y --no-install-recommends python3-setuptools
RUN apt-get install -y --no-install-recommends python3-dev
RUN apt-get install -y --no-install-recommends git
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas \
 && pip3 install numpy \
 && pip3 install boto3

RUN pip3 install --no-cache-dir flask
RUN pip3 install --no-cache-dir gevent
RUN pip3 install --no-cache-dir gunicorn
RUN pip3 install --no-cache-dir marisa_trie


RUN rm -rf /root/.cache


ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.

COPY /kggraph /opt/ml/code
COPY /info /opt/ml/info

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "serve"]
