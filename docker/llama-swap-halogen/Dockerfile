# Author: Perijn
# Summary: Packages llama-swap, Laurent llama.cpp, and Halogen so one router controls every local model lifecycle.
# Usage: Docker Compose builds this image for the llama-swap service; model commands invoke the bundled servers as supervised child processes.
FROM golang:1.27-bookworm AS llama_swap_source

ARG LLAMA_SWAP_REF=7761aa13360ea379cb89366d07c2d08aa9f1ed10
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/mostlygeek/llama-swap.git /src \
    && git -C /src checkout "${LLAMA_SWAP_REF}"

FROM node:24-bookworm AS llama_swap_ui_build

COPY --from=llama_swap_source /src /src
RUN cd /src/ui \
    && npm ci \
    && npm run build

FROM llama_swap_source AS llama_swap_build

COPY --from=llama_swap_ui_build /src/internal/server/ui_dist /src/internal/server/ui_dist
RUN cd /src \
    && go build -tags embed_ui -trimpath -o /out/llama-swap .

FROM ubuntu:24.04 AS llama_cpp_build

ARG LLAMA_CPP_REF=5e085d123eead2e89b5c19f824fccb05727da6a2
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       build-essential ca-certificates cmake git glslc libvulkan-dev spirv-headers \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/LaurentZuijdwijk/llama.cpp.git /src \
    && git -C /src checkout "${LLAMA_CPP_REF}" \
    && cmake -S /src -B /src/build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/build --target llama-server --parallel

FROM ghcr.io/peonist-ai/halogen-flash-server:0.5.6

# socat is the supervised local forwarder for the separate OCR sidecar. It lets
# llama-swap own the role/ocr lifecycle without putting Python ML runtimes in this image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       build-essential libgomp1 libvulkan1 socat \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama_swap_build /out/llama-swap /usr/local/bin/llama-swap
COPY --from=llama_cpp_build /src/build/bin/ /usr/local/bin/
COPY docker/llama-swap-halogen/scripts/halogen-launch.sh /usr/local/bin/halogen-launch
COPY docker/llama-swap-halogen/scripts/halogen-telemetry-proxy.py /usr/local/bin/halogen-telemetry-proxy
COPY docker/llama-swap-halogen/scripts/ciru-model-proxy.py /usr/local/bin/ciru-model-proxy
RUN chmod 0755 /usr/local/bin/halogen-launch /usr/local/bin/halogen-telemetry-proxy \
       /usr/local/bin/ciru-model-proxy \
    && printf '%s\n' /usr/local/bin > /etc/ld.so.conf.d/llama-server.conf \
    && ldconfig

ENTRYPOINT ["llama-swap"]