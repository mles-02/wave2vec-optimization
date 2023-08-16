FROM nvcr.io/nvidia/tritonserver:23.05-py3-sdk

WORKDIR /srv
ADD ./requirements.txt /srv/requirements.txt
RUN pip install -r requirements.txt
ADD . /srv
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

ENTRYPOINT ["bash"]