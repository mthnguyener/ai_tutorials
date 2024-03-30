#FROM nvcr.io/nvidia/pytorch:23.11-py3
#FROM nvcr.io/nvidia/pytorch:24.02-py3
FROM python:3.10-slim

ENV TORCH_HOME=/usr/src/ai_tutorials/cache

WORKDIR /usr/src/ai_tutorials

COPY . .

RUN pip install --upgrade pip \
	&& pip install -e .[all] \
	&& apt update -y \
	# && apt -y upgrade \
	&& apt install -y\
		fonts-humor-sans \
	# && conda update -y conda \
	# && while read requirement; do conda install --yes ${requirement}; done < requirements_pytorch.txt \
	# Clean up
	&& rm -rf /tmp/* \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt clean -y

CMD [ "/bin/bash" ]

