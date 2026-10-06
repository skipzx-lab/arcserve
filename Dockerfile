# arcserve on Intel Arc. The base image is used only for its Intel GPU runtime (Level Zero / OpenCL / IGC), which works
# on the Arc Pro B60; the openvino/model_server images' bundled runtime crashed on Gemma 4 for us (OVMS 2026.4.0).
FROM intel/llm-scaler-vllm:0.26.0-b2
RUN python3 -m venv /opt/arc && /opt/arc/bin/pip install -q -U pip && \
    /opt/arc/bin/pip install -q "openvino==2026.4.1" "openvino-genai==2026.4.1.0" "openvino-tokenizers==2026.4.1.0" \
      "transformers==5.5.4" "fastapi==0.142.2" "uvicorn==0.54.0" "numpy==2.4.6" "jinja2==3.1.6"
COPY arcserve.py toolparse.py /opt/arcserve/
ENV PATH=/opt/arc/bin:$PATH
WORKDIR /opt/arcserve
ENTRYPOINT []
CMD ["python", "/opt/arcserve/arcserve.py"]
